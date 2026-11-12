import asyncio

import numpy as np
from conftest import (
    LISTEN_UPDATE,
    WORDS,
    FakeTranscriber,
    append,
    running,
    send,
    session,
    silence,
    tone,
    until,
)

SPEECH_EVENTS = [
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.committed",
    "conversation.item.input_audio_transcription.completed",
]


def test_a_turn_of_speech_is_detected_and_transcribed():
    async def scenario():
        transcriber = FakeTranscriber()
        async with running(transcriber=transcriber) as url, session(url, LISTEN_UPDATE) as ws:
            await append(ws, np.concatenate([silence(0.5), tone(1.0), silence(1.0)]))
            seen = await until(ws, "conversation.item.input_audio_transcription.completed")
            # Classic voice commits after its trailing silence, when the turn has closed.
            await send(ws, {"type": "input_audio_buffer.commit"})
            refused = (await until(ws, "error"))[-1]
        return transcriber, seen, refused

    transcriber, seen, refused = asyncio.run(scenario())
    events = [event for event in seen if event["type"] in SPEECH_EVENTS]
    assert [event["type"] for event in events] == SPEECH_EVENTS
    started, stopped, _, completed = events
    assert len({event["item_id"] for event in events}) == 1
    assert completed["transcript"] == WORDS
    # The speech is placed where it was, its start padded a little.
    assert 0 <= started["audio_start_ms"] <= 500
    assert 1450 <= stopped["audio_end_ms"] <= 1600
    # The model heard the second of speech, at 16 kHz, and not the whole stream.
    (heard,) = transcriber.heard
    assert 1.0 <= len(heard) / 16000 < 2.0
    assert refused["error"]["code"] == "input_audio_buffer_commit_empty"


def test_a_commit_ends_the_speech_in_progress():
    async def scenario():
        async with running() as url, session(url, LISTEN_UPDATE) as ws:
            await append(ws, np.concatenate([silence(0.3), tone(1.0)]))
            await send(ws, {"type": "input_audio_buffer.commit"})
            return await until(ws, "conversation.item.input_audio_transcription.completed")

    seen = asyncio.run(scenario())
    assert [event["type"] for event in seen if event["type"] in SPEECH_EVENTS] == SPEECH_EVENTS
    assert seen[-1]["transcript"] == WORDS
