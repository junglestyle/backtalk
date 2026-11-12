"""Fakes for the models and the speak worker, and a client for a server on a real socket."""

import asyncio
import base64
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
from websockets.asyncio.client import connect

from backtalk.server import listening
from backtalk.session import Voice
from backtalk.speak import SpeakWorker

FAKE_SPEAK = Path(__file__).with_name("fake_speak.py")
WORDS = "turn on the lights"
# Conduit's Classic listening socket's session.update.
LISTEN_UPDATE = {
    "type": "session.update",
    "session": {
        "type": "realtime",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "turn_detection": {
                    "type": "server_vad",
                    "create_response": False,
                    "interrupt_response": False,
                },
            }
        },
    },
}


class FakeTranscriber:
    """Hears every turn as the same words, and keeps what it heard."""

    def __init__(self):
        self.heard = []

    def transcribe(self, audio):
        self.heard.append(audio)
        return WORDS


class LoudnessDetector:
    """Calls a frame speech when it's loud."""

    def scorer(self):
        return self

    def score(self, frame):
        return 1.0 if np.abs(frame).mean() > 0.05 else 0.0


@contextlib.asynccontextmanager
async def running(key=None, transcriber=None):
    """A server on a free local port; yields its URL."""
    worker = SpeakWorker([sys.executable, str(FAKE_SPEAK)])
    voice = Voice("fake", transcriber or FakeTranscriber(), LoudnessDetector(), worker)
    try:
        async with listening(voice, "127.0.0.1", 0, key) as server:
            port = next(iter(server.sockets)).getsockname()[1]
            yield f"ws://127.0.0.1:{port}/v1/realtime"
    finally:
        await worker.close()


@contextlib.asynccontextmanager
async def session(url, update=None, key=None):
    """A connected session, past session.created (and session.updated for `update`)."""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    async with connect(url, additional_headers=headers) as ws:
        await until(ws, "session.created")
        if update is not None:
            await send(ws, update)
            await until(ws, "session.updated")
        yield ws


async def send(ws, event):
    await ws.send(json.dumps(event))


async def until(ws, kind):
    """Events up to and including the first of `kind`."""
    seen = []
    while True:
        event = json.loads(await asyncio.wait_for(ws.recv(), 10))
        seen.append(event)
        if event["type"] == kind:
            return seen


def tone(seconds, rate=24000):
    t = np.arange(int(seconds * rate)) / rate
    return (np.sin(2 * np.pi * 440 * t) * 16000).astype("<i2")


def silence(seconds, rate=24000):
    return np.zeros(int(seconds * rate), dtype="<i2")


async def append(ws, pcm, rate=24000):
    """Sends `pcm` in 100 ms appends, as a phone streams its microphone."""
    step = rate // 10
    for start in range(0, len(pcm), step):
        audio = base64.b64encode(pcm[start : start + step].tobytes()).decode()
        await send(ws, {"type": "input_audio_buffer.append", "audio": audio})


def read_back(text):
    """conduit_push's read-back sentence: an out-of-band response with the text inline."""
    item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
    return {"type": "response.create", "response": {"conversation": "none", "input": [item]}}
