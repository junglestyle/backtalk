"""One connection's realtime session: OpenAI Realtime events in, events out.

backtalk is a read-back server. It hears the session's input audio, says
where speech starts and stops, and transcribes each turn; and it reads aloud
the text a response.create gives it, in Chimera's voice. It has no language
model, so it never answers on its own. conduit_push (the Hermes plugin
Conduit's calls go through) runs the turns between the two.
"""

import asyncio
import json
import sys
import time
import uuid
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from backtalk import audio, settings
from backtalk.settings import RequestError, Settings
from backtalk.speak import SpeakError, Speech
from backtalk.vad import FRAME, Segmenter, SpeechStarted, SpeechStopped

# Output audio goes out in deltas of this much speech.
DELTA_MS = 200


class Transcriber(Protocol):
    def transcribe(self, audio: audio.Samples) -> str:
        """The words in 16 kHz speech; blocking, so it runs off the event loop."""
        ...


class Scorer(Protocol):
    def score(self, frame: audio.Samples) -> float:
        """The chance a 32 ms frame at 16 kHz is speech, given the frames before it."""
        ...


class Detector(Protocol):
    def scorer(self) -> Scorer: ...


class Speaker(Protocol):
    async def speak(self, text: str) -> Speech: ...


@dataclass(frozen=True)
class Voice:
    """What every session shares: the models and the speak worker."""

    model: str
    transcriber: Transcriber
    detector: Detector
    speaker: Speaker


class _Response:
    """A response in progress: `run` reads it back, as a task of its own."""

    def __init__(self, response_id: str, run: Callable[[_Response], Coroutine[Any, Any, None]]) -> None:
        self.id = response_id
        # Set once the response is ending: nothing can cancel it any more.
        self.ending = False
        self.task = asyncio.create_task(run(self))


class Session:
    """The session for one connection. `receive` takes each client event in
    order; events go back through `send`."""

    def __init__(self, voice: Voice, send: Callable[[str], Awaitable[None]]) -> None:
        self._voice = voice
        self._send = send
        self._id = _new_id("sess")
        self._settings = Settings()
        self._to_model_rate = audio.ToModelRate(self._settings.input_rate)
        self._unframed = np.zeros(0, dtype=np.float32)
        self._scorer = voice.detector.scorer()
        self._segmenter = Segmenter(self._settings.turn_detection)
        # The item of the speech in progress, and the last one committed.
        self._item: str | None = None
        self._committed: str | None = None
        self._turns: asyncio.Queue[tuple[str, audio.Samples]] = asyncio.Queue()
        self._transcribing: asyncio.Task[None] | None = None
        # The newest user message's text, which a response.create without
        # input reads back.
        self._user_text = ""
        self._response: _Response | None = None

    async def start(self) -> None:
        self._transcribing = asyncio.create_task(self._transcribe_turns())
        await self._emit("session.created", session=self._described())

    async def close(self) -> None:
        tasks = [task for task in (self._transcribing, self._response and self._response.task) if task]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def receive(self, message: str | bytes) -> None:
        try:
            event = _event(message)
        except RequestError as error:
            await self._refuse(error, None)
            return
        try:
            await self._handle(event)
        except RequestError as error:
            await self._refuse(error, event.get("event_id"))

    async def _refuse(self, error: RequestError, event_id: Any) -> None:
        await self._emit(
            "error",
            error={
                "type": "invalid_request_error",
                "code": error.code,
                "message": error.message,
                "param": error.param,
                "event_id": event_id,
            },
        )

    async def _handle(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        match kind:
            case "session.update":
                await self._update(event.get("session"))
            case "input_audio_buffer.append":
                await self._append(event.get("audio"))
            case "input_audio_buffer.commit":
                await self._commit()
            case "input_audio_buffer.clear":
                self._segmenter.clear()
                self._unframed = np.zeros(0, dtype=np.float32)
                self._item = None
                await self._emit("input_audio_buffer.cleared")
            case "conversation.item.create":
                # Only a user message's words matter here: there's no model
                # to show the rest to.
                text = _user_text(event.get("item"))
                if text is not None:
                    self._user_text = text
            case "response.create":
                await self._create_response(event.get("response"))
            case "response.cancel":
                await self._cancel(event.get("response_id"), "client_cancelled")
            case _:
                raise RequestError("unknown_event", f"backtalk doesn't handle {kind!r} events", "type")

    # Settings.

    async def _update(self, session: Any) -> None:
        new = settings.updated(self._settings, session)
        if new.input_rate != self._settings.input_rate:
            self._to_model_rate = audio.ToModelRate(new.input_rate)
        self._segmenter.configure(new.turn_detection)
        self._settings = new
        await self._emit("session.updated", session=self._described())

    def _described(self) -> dict[str, Any]:
        return settings.described(self._settings, self._id, self._voice.model)

    # Input audio: speech detection and transcription.

    async def _append(self, data: Any) -> None:
        if not isinstance(data, str):
            raise RequestError("invalid_type", "audio must be base64 PCM16", "audio")
        try:
            pcm = audio.decode(data)
        except ValueError as error:
            raise RequestError("invalid_value", str(error), "audio") from None
        samples = np.concatenate([self._unframed, self._to_model_rate(pcm)])
        whole = len(samples) - len(samples) % FRAME
        self._unframed = samples[whole:]
        for start in range(0, whole, FRAME):
            frame = samples[start : start + FRAME]
            match self._segmenter.feed(frame, self._scorer.score(frame)):
                case SpeechStarted(start_ms=start_ms):
                    await self._speech_started(start_ms)
                case SpeechStopped() as stopped:
                    await self._speech_stopped(stopped)
                case None:
                    pass

    async def _commit(self) -> None:
        stopped = self._segmenter.cut()
        if stopped is None:
            # Conduit's Classic voice commits after its trailing silence,
            # when the turn has usually closed already; it expects this.
            raise RequestError("input_audio_buffer_commit_empty", "there's no speech in the buffer to commit")
        await self._speech_stopped(stopped)

    async def _speech_started(self, start_ms: int) -> None:
        self._item = _new_id("item")
        await self._emit("input_audio_buffer.speech_started", audio_start_ms=start_ms, item_id=self._item)
        response = self._response
        if response is not None and not response.ending and self._settings.turn_detection.interrupt_response:
            await self._cancel(response.id, "turn_detected")

    async def _speech_stopped(self, stopped: SpeechStopped) -> None:
        item = self._item or _new_id("item")
        self._item = None
        await self._emit("input_audio_buffer.speech_stopped", audio_end_ms=stopped.end_ms, item_id=item)
        await self._emit("input_audio_buffer.committed", previous_item_id=self._committed, item_id=item)
        self._committed = item
        self._turns.put_nowait((item, stopped.audio))

    async def _transcribe_turns(self) -> None:
        """Transcribes committed turns one after another, so their
        transcriptions come back in order."""
        while True:
            item, speech = await self._turns.get()
            started = time.monotonic()
            try:
                text = await asyncio.to_thread(self._voice.transcriber.transcribe, speech)
            except Exception as error:  # noqa: BLE001 — a turn that fails is reported, and the next one goes on
                print(f"transcription failed: {error!r}", file=sys.stderr)
                await self._emit(
                    "conversation.item.input_audio_transcription.failed",
                    item_id=item,
                    content_index=0,
                    error={"type": "server_error", "code": "transcription_failed", "message": str(error)},
                )
                continue
            seconds = len(speech) / audio.MODEL_RATE
            print(
                f"transcribed {seconds:.1f} s of speech in {time.monotonic() - started:.3f} s",
                file=sys.stderr,
            )
            await self._emit(
                "conversation.item.input_audio_transcription.completed",
                item_id=item,
                content_index=0,
                transcript=text,
            )

    # Responses: text read back.

    async def _create_response(self, request: Any) -> None:
        if self._response is not None:
            raise RequestError(
                "conversation_already_has_active_response",
                f"response {self._response.id} is still in progress",
            )
        text = self._response_text(request)
        response_id = _new_id("resp")
        await self._emit("response.created", response=_described_response(response_id, "in_progress"))
        self._response = _Response(response_id, lambda response: self._read_back(response, text))

    def _response_text(self, request: Any) -> str:
        """The words a response reads: its own input's user messages, or else
        the conversation's newest one."""
        if request is None:
            request = {}
        if not isinstance(request, dict):
            raise RequestError("invalid_type", "response must be an object", "response")
        items = request.get("input")
        if items is None:
            text = self._user_text
        elif isinstance(items, list):
            text = " ".join(words for words in (_user_text(item) for item in items) if words)
        else:
            raise RequestError("invalid_type", "response.input must be a list of items", "response.input")
        if not text.strip():
            raise RequestError("missing_text", "there's no user message text to read back", "response.input")
        return text.strip()

    async def _read_back(self, response: _Response, text: str) -> None:
        """Says `text` as the response's audio, then ends the response. A
        cancelled one is ended by `_cancel`."""
        try:
            started = time.monotonic()
            speech = await self._voice.speaker.speak(text)
            print(f"spoke {len(text)} characters in {time.monotonic() - started:.3f} s", file=sys.stderr)
        except SpeakError as error:
            print(str(error), file=sys.stderr)
            error_details = {"type": "server_error", "code": "speak_failed", "message": str(error)}
            await self._end(response, "failed", {"type": "failed", "error": error_details})
            return
        await self._stream(response.id, text, speech)
        await self._end(response, "completed", output=[_output_item(response.id, text)])

    async def _end(
        self,
        response: _Response,
        status: str,
        details: dict[str, Any] | None = None,
        output: list[Any] | None = None,
    ) -> None:
        response.ending = True
        self._response = None
        done = _described_response(response.id, status)
        if details is not None:
            done["status_details"] = details
        if output is not None:
            done["output"] = output
        await self._emit("response.done", response=done)

    async def _stream(self, response_id: str, text: str, speech: Speech) -> None:
        rate = self._settings.output_rate
        pcm = audio.resampled(speech.pcm, speech.rate, rate)
        ids = {
            "response_id": response_id,
            "item_id": _item_id(response_id),
            "output_index": 0,
            "content_index": 0,
        }
        await self._emit("response.output_audio_transcript.delta", **ids, delta=text)
        step = rate * DELTA_MS // 1000
        for start in range(0, len(pcm), step):
            await self._emit(
                "response.output_audio.delta", **ids, delta=audio.encode(pcm[start : start + step])
            )
        await self._emit("response.output_audio.done", **ids)
        await self._emit("response.output_audio_transcript.done", **ids, transcript=text)

    async def _cancel(self, response_id: Any, reason: str) -> None:
        response = self._response
        if response is None or response.ending or response_id not in (None, response.id):
            raise RequestError("response_cancel_not_active", "there's no response in progress to cancel")
        response.task.cancel()
        await asyncio.gather(response.task, return_exceptions=True)
        # Ended before the next client event is handled, and nothing of it
        # goes out after this.
        await self._end(response, "cancelled", {"type": "cancelled", "reason": reason})

    async def _emit(self, kind: str, **fields: Any) -> None:
        await self._send(json.dumps({"type": kind, "event_id": _new_id("event"), **fields}))


def _event(message: str | bytes) -> dict[str, Any]:
    if not isinstance(message, str):
        raise RequestError("invalid_json", "events are JSON text frames, not binary")
    try:
        event = json.loads(message)
    except ValueError:
        raise RequestError("invalid_json", "the event isn't JSON") from None
    if not isinstance(event, dict):
        raise RequestError("invalid_json", "an event is a JSON object")
    return event


def _user_text(item: Any) -> str | None:
    """A user message item's text parts, joined; None for any other item."""
    if not isinstance(item, dict) or item.get("role") != "user":
        return None
    content = item.get("content")
    if not isinstance(content, list):
        return None
    parts = [
        part["text"]
        for part in content
        if isinstance(part, dict)
        and part.get("type") in ("input_text", "text")
        and isinstance(part.get("text"), str)
    ]
    return " ".join(parts)


def _described_response(response_id: str, status: str) -> dict[str, Any]:
    return {"id": response_id, "object": "realtime.response", "status": status, "output": []}


def _item_id(response_id: str) -> str:
    return "item_" + response_id.removeprefix("resp_")


def _output_item(response_id: str, text: str) -> dict[str, Any]:
    return {
        "id": _item_id(response_id),
        "object": "realtime.item",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_audio", "transcript": text}],
    }


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:20]}"
