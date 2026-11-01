"""The web keyboard's Web Audio synth (keyboard.js) must sound like the Python one.

Renders the same notes in headless Chrome with an OfflineAudioContext and compares the
result with web.render.render. Skipped without Chrome.
"""

import http.server
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import urllib.parse

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from musicfit.loss import MultiResolutionSTFTLoss  # noqa: E402
from musicfit.notes import Note  # noqa: E402
from musicfit.synth import PARAM_SPECS  # noqa: E402

pytest.importorskip("fastapi")
from musicfit.web.render import render  # noqa: E402

CHROME = next((c for c in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser") if shutil.which(c)), None)
SR = 16000
SECONDS = 2.5
KEYBOARD_JS = os.path.join(os.path.dirname(__file__), "..", "musicfit", "web", "static", "keyboard.js")
NOTES = [Note(48.0, 0.1, 0.5, 0.9), Note(60.0, 0.7, 0.4, 0.6), Note(67.0, 0.7, 0.4, 0.6), Note(55.0, 1.3, 0.6, 1.0)]

BASE = {n: s.default for n, s in PARAM_SPECS.items()}
BASE.update(waveform={"sine": 0.1, "saw": 0.6, "square": 0.2, "triangle": 0.1}, delay_mix=0.0, reverb_mix=0.0, noise_level=0.0)
PATCHES = {
    "plain": {},
    "bright": {"cutoff": 6000.0, "filt_env_amount": 0.0, "waveform": {"sine": 0.0, "saw": 1.0, "square": 0.0, "triangle": 0.0}},
    "dark and resonant": {"cutoff": 300.0, "resonance": 6.0, "filt_env_amount": 3.0, "filt_decay": 0.15, "amp_sustain": 0.2},
    "square": {"waveform": {"sine": 0.0, "saw": 0.0, "square": 1.0, "triangle": 0.0}},
    "slow envelopes": {"amp_attack": 0.2, "amp_decay": 0.5, "amp_release": 0.5, "filt_attack": 0.3, "filt_release": 0.5},
    "driven": {"drive": 5.0, "gain": 0.3},
    "echoes": {"delay_mix": 0.6, "delay_feedback": 0.5, "delay_time": 0.3},
    "reverb": {"reverb_mix": 0.5, "reverb_rt60": 1.2},
    "noise": {"noise_level": 0.3},
    "unison": {"unison_cents": 20.0},
}


PAGE = """<!doctype html><body><script>%(script)s</script><script>
(async () => {
  for (const [name, patch] of Object.entries(%(patches)s)) {
    const ctx = new OfflineAudioContext(1, %(samples)d, %(sr)d);
    const synth = new LiveSynth(ctx);
    synth.setPatch(patch, 32);
    for (const n of %(notes)s) synth.noteOff(synth.noteOn(n.pitch, n.velocity, n.onset), n.onset + n.duration);
    const buffer = await ctx.startRendering();
    await fetch("/result/" + encodeURIComponent(name), { method: "POST", body: buffer.getChannelData(0).buffer });
  }
})().catch((e) => fetch("/error", { method: "POST", body: String(e) }));
</script></body>"""


@pytest.fixture(scope="module")
def chrome_renders(tmp_path_factory):
    """{patch name: what keyboard.js renders for it}, posted back by the page to a local server."""
    with open(KEYBOARD_JS) as f:
        script = f.read()
    patches = {name: {**BASE, **changes} for name, changes in PATCHES.items()}
    notes = [{"pitch": n.pitch, "onset": n.onset, "duration": n.duration, "velocity": n.velocity} for n in NOTES]
    page = PAGE % {"script": script, "patches": json.dumps(patches), "samples": int(SECONDS * SR), "sr": SR, "notes": json.dumps(notes)}
    results = queue.Queue()

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = page.encode() if self.path == "/" else b""
            self.send_response(200 if body else 404)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            results.put((self.path, self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    chrome = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--no-sandbox", "--remote-debugging-port=0",
                               f"--user-data-dir={tmp_path_factory.mktemp('chrome')}", f"http://127.0.0.1:{server.server_port}/"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = {}
    try:
        while len(out) < len(patches):
            path, data = results.get(timeout=120)
            assert path.startswith("/result/"), data
            out[urllib.parse.unquote(path[len("/result/"):])] = torch.from_numpy(np.frombuffer(data, dtype=np.float32).copy())
    finally:
        chrome.kill()
        chrome.wait()
        server.shutdown()
    return out


def _python(patch: dict) -> torch.Tensor:
    return render([patch], NOTES, int(SECONDS * SR), SR, 32)


@pytest.mark.skipif(CHROME is None, reason="needs Chrome")
@pytest.mark.parametrize("name", PATCHES)
def test_live_synth_matches_the_python_synth(name, chrome_renders):
    patch = {**BASE, **PATCHES[name]}
    live, reference = chrome_renders[name], _python(patch)
    loss = MultiResolutionSTFTLoss()
    distance = float(loss(live, reference))
    level = float(live.pow(2).mean().sqrt() / reference.pow(2).mean().sqrt())
    print(f"{name}: distance {distance:.3f}, level ratio {level:.3f}")
    if name == "reverb":
        # the impulse response is random noise, and its level at the note's few harmonics varies
        # as much between seeds in Python alone (0.85 to 1.31 times), so only the level can be held to
        assert 0.65 < level < 1.55
        return
    assert 0.8 < level < 1.25
    # a different patch of the same notes is far further away than this
    other = _python({**BASE, "cutoff": 400.0 if patch["cutoff"] > 800 else 4000.0, "amp_sustain": 0.1})
    assert distance < 0.6 * float(loss(other, reference))
