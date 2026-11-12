import asyncio

import numpy as np
import pytest
from conftest import LISTEN_UPDATE, WORDS, append, read_back, running, send, session, silence, tone, until
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

KEY = "test-key-0000"


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong-key"}, {"Authorization": KEY}])
def test_a_connection_without_the_key_is_refused(headers):
    async def scenario():
        async with running(key=KEY) as url:
            with pytest.raises(InvalidStatus) as refused:
                async with connect(url, additional_headers=headers):
                    pass
            # The right key still gets in.
            async with session(url, key=KEY):
                pass
        return refused.value.response.status_code

    assert asyncio.run(scenario()) == 401


def test_a_listening_and_a_speaking_socket_work_at_once():
    async def scenario():
        async with running() as url, session(url, LISTEN_UPDATE) as listening, session(url) as speaking:
            await send(speaking, read_back("[slow] One two three."))
            await append(listening, np.concatenate([tone(1.0), silence(1.0)]))
            return await asyncio.gather(
                until(listening, "conversation.item.input_audio_transcription.completed"),
                until(speaking, "response.done"),
            )

    heard, spoken = asyncio.run(scenario())
    assert heard[-1]["transcript"] == WORDS
    assert spoken[-1]["response"]["status"] == "completed"
