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


def _run_page_in_chrome(page: str, n_results: int, profile) -> dict:
    """Open a page in headless Chrome and return {path: body} for the n_results it POSTs back."""
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
                               f"--user-data-dir={profile}", f"http://127.0.0.1:{server.server_port}/"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = {}
    try:
        while len(out) < n_results:
            path, data = results.get(timeout=120)
            assert path != "/error", data
            out[path] = data
    finally:
        chrome.kill()
        chrome.wait()
        server.shutdown()
    return out


@pytest.fixture(scope="module")
def chrome_renders(tmp_path_factory):
    """{patch name: what keyboard.js renders for it}."""
    with open(KEYBOARD_JS) as f:
        script = f.read()
    patches = {name: {**BASE, **changes} for name, changes in PATCHES.items()}
    notes = [{"pitch": n.pitch, "onset": n.onset, "duration": n.duration, "velocity": n.velocity} for n in NOTES]
    page = PAGE % {"script": script, "patches": json.dumps(patches), "samples": int(SECONDS * SR), "sr": SR, "notes": json.dumps(notes)}
    posted = _run_page_in_chrome(page, len(patches), tmp_path_factory.mktemp("chrome"))
    return {urllib.parse.unquote(path[len("/result/"):]): torch.from_numpy(np.frombuffer(data, dtype=np.float32).copy())
            for path, data in posted.items()}


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


# The real page's markup with keyboard.js alone, driven by key events aimed where focus is.
KEYS_TEST = """
for (const id of ["job-view", "result-view"]) document.getElementById(id).hidden = false;
const textBox = document.createElement("input");
textBox.type = "text";
document.querySelector(".card.play").appendChild(textBox);
const played = [];
const noteOn = LiveSynth.prototype.noteOn;
LiveSynth.prototype.noteOn = function (pitch, velocity, when, slide) { played.push(pitch); return noteOn.call(this, pitch, velocity, when, slide); };
const realContext = window.AudioContext;
const keyboard = setupKeyboard({ getPatch: () => (%(patch)s), getHarmonics: () => 32, getSampleRate: () => 16000, isActive: () => true,
                                 noteName: (p) => "n" + p, defaultSlide: () => false });
const fire = (type, code, key) => document.activeElement.dispatchEvent(new KeyboardEvent(type, { code, key, bubbles: true, cancelable: true }));
const lit = () => document.querySelectorAll(".pkey.down").length;
const status = () => document.querySelector("#kbd-message").textContent;
function tap(code = "KeyA", key = "a") {
  const before = played.length;
  fire("keydown", code, key);
  const out = { played: played.length - before, lit: lit(), status: status(), pressed: document.querySelector("#kbd-pressed").textContent };
  fire("keyup", code, key);
  out.litAfter = lit();
  return out;
}
const focusOn = (el) => { el.focus(); if (document.activeElement !== el) throw new Error("couldn't focus " + el.outerHTML); };
const out = {};
window.AudioContext = function () { throw new Error("no audio device"); };
out.noAudio = { ...tap(), sound: document.querySelector("#kbd-sound").textContent };
window.AudioContext = realContext;
out.body = tap();
focusOn(document.querySelector("[name=duration]")); out.numberBox = tap();
focusOn(document.querySelector("#track")); out.dropdown = tap();
focusOn(document.querySelector("#kbd-on")); out.checkbox = tap();
focusOn(textBox);
textBox.dispatchEvent(new FocusEvent("focusin", { bubbles: true }));  // a headless page without window focus gets none
out.textBoxBefore = status(); out.textBox = tap();
fire("keydown", "Escape", "Escape"); out.afterEscape = { active: document.activeElement.tagName, ...tap() };
out.noPhysicalKey = tap("", "a");
out.notAKey = tap("KeyQ", "q");
const m = fire("keydown", "KeyM", "m"); fire("keyup", "KeyM", "m");
out.off = tap();
fire("keydown", "KeyM", "m"); fire("keyup", "KeyM", "m");
const shortcut = new KeyboardEvent("keydown", { code: "KeyA", key: "a", ctrlKey: true, bubbles: true, cancelable: true });
const before = played.length; document.body.dispatchEvent(shortcut); out.ctrlPlayed = played.length - before; out.ctrlDefaultKept = !shortcut.defaultPrevented;
fetch("/keys", { method: "POST", body: JSON.stringify(out) });
"""


@pytest.fixture(scope="module")
def key_results(tmp_path_factory):
    static = os.path.dirname(KEYBOARD_JS)
    with open(os.path.join(static, "index.html")) as f:
        html = f.read()
    with open(KEYBOARD_JS) as f:
        script = f.read()
    # just the page's elements: no stylesheet, and app.js would need a server
    html = html.replace('<script src="keyboard.js"></script>', "").replace('<script src="app.js"></script>', "")
    test = "(async () => {" + KEYS_TEST % {"patch": json.dumps(BASE)} + "})().catch((e) => fetch('/error', { method: 'POST', body: String(e) }));"
    page = html.replace("</body>", f"<script>{script}</script><script>{test}</script></body>")
    return json.loads(_run_page_in_chrome(page, 1, tmp_path_factory.mktemp("chrome"))["/keys"])


needs_chrome = pytest.mark.skipif(CHROME is None, reason="needs Chrome")


@needs_chrome
@pytest.mark.parametrize("where", ["body", "numberBox", "dropdown", "checkbox"])
def test_keys_play_wherever_focus_has_no_use_for_letters(key_results, where):
    r = key_results[where]
    assert r["played"] == 1 and r["lit"] == 1 and r["litAfter"] == 0
    assert "A" in r["pressed"] and "n60" in r["pressed"]


@needs_chrome
def test_a_text_box_keeps_the_keys_and_says_so(key_results):
    assert "text box" in key_results["textBoxBefore"]
    r = key_results["textBox"]
    assert r["played"] == 0 and r["lit"] == 0 and "didn't play" in r["status"]


@needs_chrome
def test_escape_leaves_a_text_box(key_results):
    r = key_results["afterEscape"]
    assert r["active"] == "BODY" and r["played"] == 1


@needs_chrome
def test_a_key_lights_up_even_if_sound_fails(key_results):
    r = key_results["noAudio"]
    assert r["lit"] == 1 and "A" in r["pressed"] and "no audio device" in r["sound"]


@needs_chrome
def test_keys_without_a_physical_code_still_play(key_results):
    assert key_results["noPhysicalKey"]["played"] == 1


@needs_chrome
def test_other_keys_and_the_off_switch_explain_themselves(key_results):
    assert "isn't a key" in key_results["notAKey"]["status"]
    off = key_results["off"]
    assert off["played"] == 0 and "off" in off["status"]


@needs_chrome
def test_browser_shortcuts_are_left_alone(key_results):
    assert key_results["ctrlPlayed"] == 0 and key_results["ctrlDefaultKept"]
