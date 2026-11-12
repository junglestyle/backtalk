"""The realtime WebSocket endpoint: one session per connection, many at once."""

import hmac
import sys
import urllib.parse

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request, Response

from backtalk.session import Session, Voice

PATH = "/v1/realtime"
# Conduit sends at most a second of 24 kHz audio an event.
MAX_EVENT_BYTES = 1024 * 1024


def listening(voice: Voice, host: str, port: int, key: str | None) -> Server:
    """The server, serving once entered (`async with`). With `key`, a
    connection without `Authorization: Bearer <key>` is refused with 401."""

    def check(connection: ServerConnection, request: Request) -> Response | None:
        if urllib.parse.urlsplit(request.path).path != PATH:
            return connection.respond(404, f"backtalk serves {PATH}\n")
        if key is not None:
            given = request.headers.get("Authorization", "")
            if not hmac.compare_digest(given.encode(), f"Bearer {key}".encode()):
                return connection.respond(401, "a bearer key is required\n")
        return None

    async def run_session(connection: ServerConnection) -> None:
        session = Session(voice, connection.send)
        peer = connection.remote_address
        print(f"session from {peer} opened", file=sys.stderr)
        try:
            await session.start()
            async for message in connection:
                await session.receive(message)
        except ConnectionClosed:
            pass
        finally:
            await session.close()
            print(f"session from {peer} closed", file=sys.stderr)

    return serve(run_session, host, port, process_request=check, max_size=MAX_EVENT_BYTES)
