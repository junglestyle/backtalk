import asyncio
import base64
import json

import pytest
from conftest import append, read_back, running, send, session, tone, until

SENTENCE = "Once upon a time, there was a fox."
# conduit_push's read-back session.update: the call's own, with automatic responses off.
READ_BACK_UPDATE = {
    "type": "session.update",
    "session": {
        "type": "realtime",
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "turn_detection": {
                    "type": "server_vad",
                    "create_response": False,
                    "interrupt_response": False,
                },
            },
            "output": {"format": {"type": "audio/pcm", "rate": 24000}},
        },
    },
}


def _output_rate(rate):
    update = json.loads(json.dumps(READ_BACK_UPDATE))
    update["session"]["audio"]["output"]["format"]["rate"] = rate
    return update


def _classic(text):
    """Classic voice's speak socket: a user item, then a bare response.create."""
    item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
    return [{"type": "conversation.item.create", "item": item}, {"type": "response.create"}]


@pytest.mark.parametrize(("frames", "rate"), [([read_back(SENTENCE)], 24000), (_classic(SENTENCE), 16000)])
def test_a_response_reads_its_text_back_at_the_sessions_rate(frames, rate):
    async def scenario():
        async with running() as url, session(url, _output_rate(rate)) as ws:
            for frame in frames:
                await send(ws, frame)
            return await until(ws, "response.done")

    seen = asyncio.run(scenario())
    kinds = [event["type"] for event in seen]
    deltas = kinds.count("response.output_audio.delta")
    assert kinds == [
        "response.created",
        "response.output_audio_transcript.delta",
        *["response.output_audio.delta"] * deltas,
        "response.output_audio.done",
        "response.output_audio_transcript.done",
        "response.done",
    ]
    assert seen[-2]["transcript"] == SENTENCE
    assert seen[-1]["response"]["status"] == "completed"
    assert len({event.get("response_id") or event["response"]["id"] for event in seen}) == 1
    samples = sum(
        len(base64.b64decode(e["delta"])) // 2 for e in seen if e["type"] == "response.output_audio.delta"
    )
    # The fake worker says 0.1 s a word, at 48 kHz.
    assert samples == pytest.approx(0.8 * rate, rel=0.01)


@pytest.mark.parametrize(
    "turn_detection", [{"type": "server_vad", "create_response": True}, {"type": "server_vad"}]
)
def test_automatic_responses_are_refused_as_the_plugin_expects(turn_detection):
    asked = json.loads(json.dumps(READ_BACK_UPDATE))
    asked["session"]["audio"]["input"]["turn_detection"] = turn_detection

    async def scenario():
        async with running() as url, session(url) as ws:
            await send(ws, asked)
            refusal = await until(ws, "error")
            # The plugin sends the update again with automatic responses off.
            await send(ws, READ_BACK_UPDATE)
            return refusal, await until(ws, "session.updated")

    refusal, updated = asyncio.run(scenario())
    assert [event["type"] for event in refusal] == ["error"]
    assert refusal[0]["error"]["code"] == "realtime_live_unavailable"
    assert refusal[0]["error"]["param"] == "session.audio.input.turn_detection.create_response"
    assert [event["type"] for event in updated] == ["session.updated"]
    detection = updated[0]["session"]["audio"]["input"]["turn_detection"]
    assert detection["create_response"] is False
    assert detection["interrupt_response"] is False


def test_a_cancelled_response_stops_and_the_next_reads_its_own_text():
    async def scenario():
        async with running() as url, session(url, READ_BACK_UPDATE) as ws:
            await send(ws, read_back("[slow] One two three four five."))
            await until(ws, "response.created")
            # One response at a time: the plugin sends the next sentence after this one's end.
            await send(ws, read_back("Too soon."))
            busy = (await until(ws, "error"))[-1]
            await send(ws, {"type": "response.cancel"})
            cancelled = await until(ws, "response.done")
            await send(ws, read_back("Six seven eight."))
            following = await until(ws, "response.done")
            # Nothing is left to cancel; the plugin keeps that error to itself.
            await send(ws, {"type": "response.cancel"})
            nothing = (await until(ws, "error"))[-1]
        return busy, cancelled, following, nothing

    busy, cancelled, following, nothing = asyncio.run(scenario())
    assert busy["error"]["code"] == "conversation_already_has_active_response"
    assert [event["type"] for event in cancelled] == ["response.done"]
    assert cancelled[0]["response"]["status"] == "cancelled"
    assert following[-2]["transcript"] == "Six seven eight."
    assert following[-1]["response"]["status"] == "completed"
    assert "cancel" in nothing["error"]["code"]


def test_speech_over_a_response_stops_it():
    async def scenario():
        async with running() as url, session(url) as ws:
            await send(ws, read_back("[slow] One two three four five."))
            await until(ws, "response.created")
            await append(ws, tone(0.5))
            return await until(ws, "response.done")

    seen = asyncio.run(scenario())
    assert [event["type"] for event in seen] == ["input_audio_buffer.speech_started", "response.done"]
    assert seen[-1]["response"]["status"] == "cancelled"
    assert seen[-1]["response"]["status_details"]["reason"] == "turn_detected"


def test_a_speak_worker_that_dies_is_started_again():
    async def scenario():
        async with running() as url, session(url, READ_BACK_UPDATE) as ws:
            await send(ws, read_back("[die] One two three."))
            failed = await until(ws, "response.done")
            await send(ws, read_back("Four five six."))
            return failed, await until(ws, "response.done")

    failed, following = asyncio.run(scenario())
    assert [event["type"] for event in failed] == ["response.created", "response.done"]
    assert failed[-1]["response"]["status"] == "failed"
    assert following[-2]["transcript"] == "Four five six."
    assert following[-1]["response"]["status"] == "completed"
