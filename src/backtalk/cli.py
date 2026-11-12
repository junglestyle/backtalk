import argparse
import asyncio
import os
import signal
import sys
from pathlib import Path

from backtalk import models
from backtalk.server import PATH, listening
from backtalk.session import Voice
from backtalk.speak import SpeakError, SpeakWorker

HOST = "0.0.0.0"  # noqa: S104 — it runs in a container; the host publishes it on one address
PORT = 8765


def main() -> None:
    argparse.ArgumentParser(
        description=f"Serves the OpenAI Realtime protocol on ws://{HOST}:{PORT}{PATH}: it transcribes speech "
        "and reads text back in Chimera's voice. Set BACKTALK_SPEAK and BACKTALK_CACHE, and BACKTALK_API_KEY "
        "to require a key."
    ).parse_args()
    cache = os.environ.get("BACKTALK_CACHE")
    if not cache:
        raise SystemExit("BACKTALK_CACHE is not set: it's the directory the models download into")
    speak = os.environ.get("BACKTALK_SPEAK")
    if not speak:
        raise SystemExit("BACKTALK_SPEAK is not set: it's the path of the speak worker, speak/speak.mjs")
    if not Path(speak).is_file():
        raise SystemExit(f"BACKTALK_SPEAK is {speak!r}, which isn't a file")
    key = os.environ.get("BACKTALK_API_KEY")
    if key is not None and not key.strip():
        raise SystemExit("BACKTALK_API_KEY is empty: set a key, or unset it to take connections without one")
    worker = SpeakWorker(["node", speak, "--serve"])
    voice = Voice(
        model=models.MODEL,
        transcriber=models.Parakeet(Path(cache)),
        detector=models.Silero(Path(cache)),
        speaker=worker,
    )
    asyncio.run(_serve(voice, worker, key))


async def _serve(voice: Voice, worker: SpeakWorker, key: str | None) -> None:
    try:
        # Starts the worker and loads its voice before the first reply needs it.
        await worker.speak("Ready.")
    except SpeakError as error:
        raise SystemExit(f"the speak worker doesn't work: {error}") from None
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stop.set)
    async with listening(voice, HOST, PORT, key):
        print(f"listening on ws://{HOST}:{PORT}{PATH}, {'with' if key else 'without'} a key", file=sys.stderr)
        await stop.wait()
    await worker.close()
