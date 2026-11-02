"""The server keeps a record of when and from where something was changed."""

import logging
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from musicfit.web.app import create_app  # noqa: E402
from musicfit.web.request_log import RequestLog, timestamp_server_logs  # noqa: E402

LINE = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d [+-]\d{4} (\S+) (\S+) (\S+) (\d{3})$")


@pytest.fixture()
def jobs_dir(tmp_path):
    return tmp_path / "jobs"


@pytest.fixture()
def client(jobs_dir, tmp_path):
    return TestClient(create_app(str(jobs_dir), str(tmp_path / "data")), raise_server_exceptions=False)


def _lines(jobs_dir):
    path = jobs_dir / "requests.log"
    return path.read_text().splitlines() if path.exists() else []


def test_posts_are_logged_with_time_address_and_outcome(client, jobs_dir):
    assert client.post("/api/jobs", data={"duration": "600"}).status_code == 422
    lines = _lines(jobs_dir)
    assert len(lines) == 1
    assert LINE.match(lines[0]).groups() == ("testclient", "POST", "/api/jobs", "422")


def test_deletes_are_logged_and_reads_are_not(client, jobs_dir):
    client.get("/api/jobs")
    client.get("/")
    client.head("/")
    assert _lines(jobs_dir) == []
    client.delete("/api/jobs/nothing")
    assert LINE.match(_lines(jobs_dir)[0]).groups() == ("testclient", "DELETE", "/api/jobs/nothing", "404")


def test_the_log_outlasts_a_restart(client, jobs_dir, tmp_path):
    client.post("/api/midi", json={"notes": []})
    again = TestClient(create_app(str(jobs_dir), str(tmp_path / "data")))
    again.post("/api/midi", json={"notes": []})
    assert len(_lines(jobs_dir)) == 2
    assert again.get("/api/jobs").json() == []  # and the job list isn't confused by it


def test_a_client_cant_forge_a_line_with_an_encoded_newline(client, jobs_dir):
    client.post("/api/jobs/x%0A2026-01-01%2000:00:00%20-0700%201.2.3.4%20POST%20/forged%20200/cancel")
    lines = _lines(jobs_dir)
    assert len(lines) == 1 and "%0A" in lines[0]


def test_a_crash_is_logged_as_a_server_error(tmp_path):
    app = FastAPI()

    @app.post("/boom")
    def boom():
        raise RuntimeError("boom")

    path = tmp_path / "requests.log"
    app.add_middleware(RequestLog, path=str(path))
    TestClient(app, raise_server_exceptions=False).post("/boom")
    assert LINE.match(path.read_text().strip()).group(4) == "500"


def test_failing_to_write_the_log_doesnt_fail_the_request(tmp_path):
    app = FastAPI()

    @app.post("/fine")
    def fine():
        return {"ok": True}

    app.add_middleware(RequestLog, path=str(tmp_path / "no_such_directory" / "requests.log"))
    assert TestClient(app).post("/fine").status_code == 200


def test_the_terminal_lines_get_the_time(monkeypatch):
    uvicorn_logging = pytest.importorskip("uvicorn.logging")
    handler = logging.StreamHandler()
    logger = logging.getLogger("uvicorn.access")
    monkeypatch.setattr(logger, "handlers", [handler])
    timestamp_server_logs()
    record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s HTTP/%s" %d',
                               ("10.0.1.2:5000", "POST", "/api/jobs", "1.1", 200), None)
    assert isinstance(handler.formatter, uvicorn_logging.AccessFormatter)
    assert re.match(r'^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d [+-]\d{4} .*10\.0\.1\.2:5000 - "POST /api/jobs HTTP/1\.1" 200', handler.format(record))
