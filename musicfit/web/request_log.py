"""Who did what: the time and client address of requests, on the terminal and in a file that lasts."""

from __future__ import annotations

import logging
from datetime import datetime

READ_ONLY_METHODS = {"GET", "HEAD", "OPTIONS"}
TIME_FORMAT = "%Y-%m-%d %H:%M:%S %z"  # with the UTC offset, so that times stay clear across daylight saving


def _now() -> str:
    return datetime.now().astimezone().strftime(TIME_FORMAT)


class RequestLog:
    """ASGI middleware appending a line per request that changes something to a file:

        2026-11-01 01:31:28 -0700 10.0.1.2 POST /api/jobs 200

    The address is what the server sees, so behind an SSH tunnel or a proxy it is theirs. The path is
    written as it came, still percent-encoded, so a client can't forge lines with an encoded newline.
    """

    def __init__(self, app, path: str):
        self.app = app
        self.path = path

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in READ_ONLY_METHODS:
            await self.app(scope, receive, send)
            return
        arrived = _now()
        status = 500  # unless the app gets as far as answering

        async def note_status(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, note_status)
        finally:
            client = (scope.get("client") or ("unknown",))[0]
            target = (scope.get("raw_path") or scope["path"].encode()).decode("latin-1")
            self._write(f"{arrived} {client} {scope['method']} {target} {status}\n")

    def _write(self, line: str) -> None:
        try:
            with open(self.path, "a") as f:
                f.write(line)
        except OSError as e:  # a full disk shouldn't turn into failed requests
            logging.getLogger("uvicorn.error").warning("couldn't write to %s: %s", self.path, e)


def timestamp_server_logs() -> None:
    """Put the time on uvicorn's terminal lines. Its access lines already show the client's address."""
    try:
        from uvicorn.logging import AccessFormatter, DefaultFormatter
    except ImportError:  # some other server
        return
    formats = {
        "uvicorn": DefaultFormatter("%(asctime)s %(levelprefix)s %(message)s", datefmt=TIME_FORMAT),
        "uvicorn.access": AccessFormatter('%(asctime)s %(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                                          datefmt=TIME_FORMAT),
    }
    for name, formatter in formats.items():
        for handler in logging.getLogger(name).handlers:
            handler.setFormatter(formatter)
