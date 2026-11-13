"""The speak worker: Chimera's voice, as one long-lived child process.

The worker (chimera-tts's `speak.mjs --serve`) reads one JSON request
a line on stdin, `{"id", "text"}`, and answers each in order with one line on
stdout: `{"id", "rate", "pcm"}` (base64 PCM16 of the whole text) or
`{"id", "error"}`. It loads its voice once, so it stays up between requests;
one that dies or hangs is started again for the next request.
"""

import asyncio
import contextlib
import itertools
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from backtalk import audio

# A request the worker hasn't answered in this long has hung it.
TIMEOUT_S = 60.0
# A reply line carries a whole sentence's audio.
MAX_LINE_BYTES = 64 * 1024 * 1024
EXIT_WAIT_S = 5.0


class SpeakError(Exception):
    """The worker couldn't say a text."""


@dataclass(frozen=True)
class Speech:
    rate: int
    pcm: audio.Pcm16


class SpeakWorker:
    """Says texts one at a time through the worker `command` starts."""

    def __init__(self, command: Sequence[str], timeout: float = TIMEOUT_S) -> None:
        self._command = list(command)
        self._timeout = timeout
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._ids = itertools.count(1)

    async def speak(self, text: str) -> Speech:
        """`text` in Chimera's voice; SpeakError when the worker can't say it.

        A caller cancelled mid-request stops waiting, but the request runs to
        its answer first, so the next one reads its own line.
        """
        await self._lock.acquire()
        exchange = asyncio.ensure_future(self._exchange(text))
        exchange.add_done_callback(lambda _: self._lock.release())
        return await asyncio.shield(exchange)

    async def close(self) -> None:
        """Ends the worker: it exits when its stdin closes."""
        async with self._lock:
            process, self._process = self._process, None
            if process is None or process.returncode is not None:
                return
            _pipes(process)[0].close()
            try:
                await asyncio.wait_for(process.wait(), EXIT_WAIT_S)
            except TimeoutError:
                process.kill()
                await process.wait()

    async def _exchange(self, text: str) -> Speech:
        process = await self._running()
        stdin, stdout = _pipes(process)
        request_id = str(next(self._ids))
        try:
            stdin.write(json.dumps({"id": request_id, "text": text}).encode() + b"\n")
            await stdin.drain()
            line = await asyncio.wait_for(stdout.readline(), self._timeout)
        except TimeoutError:
            await self._stop(process)
            raise SpeakError(
                f"the speak worker didn't answer in {self._timeout:.0f} s; it starts again"
            ) from None
        except BrokenPipeError, ConnectionResetError, ValueError:
            # ValueError: a line over MAX_LINE_BYTES.
            await self._stop(process)
            raise SpeakError("the speak worker broke off; it starts again") from None
        if not line:
            code = await process.wait()
            self._process = None
            raise SpeakError(f"the speak worker exited with {code}; it starts again")
        try:
            return _speech(line, request_id)
        except ValueError as error:
            await self._stop(process)
            raise SpeakError(f"the speak worker's answer made no sense ({error}); it starts again") from None

    async def _running(self) -> asyncio.subprocess.Process:
        if self._process is not None and self._process.returncode is None:
            return self._process
        if self._process is not None:
            print(
                f"the speak worker exited with {self._process.returncode}; starting it again", file=sys.stderr
            )
        try:
            # Its stderr (its own diagnostics) goes to ours.
            self._process = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                limit=MAX_LINE_BYTES,
            )
        except OSError as error:
            raise SpeakError(f"couldn't start the speak worker {self._command!r}: {error}") from None
        return self._process

    async def _stop(self, process: asyncio.subprocess.Process) -> None:
        self._process = None
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        await process.wait()


def _speech(line: bytes, request_id: str) -> Speech:
    """The worker's answer to `request_id`; SpeakError for its refusal, ValueError for nonsense."""
    reply = json.loads(line)
    if not isinstance(reply, dict) or reply.get("id") != request_id:
        raise ValueError(f"expected the answer to request {request_id!r}")
    if "error" in reply:
        raise SpeakError(f"the speak worker couldn't say it: {reply['error']}")
    rate, pcm = reply.get("rate"), reply.get("pcm")
    if not isinstance(rate, int) or rate <= 0 or not isinstance(pcm, str):
        raise ValueError("expected rate and pcm")
    return Speech(rate, audio.decode(pcm))


def _pipes(process: asyncio.subprocess.Process) -> tuple[asyncio.StreamWriter, asyncio.StreamReader]:
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("the speak worker was started without pipes")
    return process.stdin, process.stdout
