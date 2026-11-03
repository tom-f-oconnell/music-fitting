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
from musicfit.drums import KIT_SPECS  # noqa: E402
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
focusOn(document.querySelector("[name=quality]")); out.dropdown = tap();
focusOn(document.querySelector("#track-search")); out.searchBox = tap(); document.activeElement.blur();
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
def test_typing_in_the_track_search_doesnt_play(key_results):
    assert key_results["searchBox"]["played"] == 0


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


# ---- the track list: the real page and script, with the server's answers filled in

LIBRARY = {
    "root": "/data",
    "datasets": [
        {"id": "examples", "name": "Examples", "source": "", "query": "", "limit": None, "size_bytes": 0, "tracks": [
            {"id": "examples/sparse_piano.flac", "title": "Sparse piano", "duration_s": None, "offset": 0.0, "length": 4.0, "voices": 1, "note": "Few notes."},
            {"id": "aphex_twin/audio/Cornish_Acid.opus", "title": "Cornish Acid", "duration_s": None, "offset": 30.0, "length": 4.0, "voices": 2, "note": "Acid."},
        ]},
        {"id": "aphex_twin", "name": "Aphex Twin", "source": "aphex_twin", "query": "Aphex Twin", "limit": None, "size_bytes": 1000, "tracks": [
            {"id": "aphex_twin/audio/Cornish_Acid.opus", "title": "Aphex Twin - Cornish Acid", "duration_s": 134.0, "offset": 0.0, "length": 4.0, "voices": 1, "note": ""},
            {"id": "aphex_twin/audio/Xtal.opus", "title": "Aphex Twin - Xtal", "duration_s": 291.0, "offset": 0.0, "length": 4.0, "voices": 1, "note": ""},
        ]},
        {"id": "other", "name": "Björk search", "source": "search", "query": "x", "limit": 5, "size_bytes": 1000, "tracks": [
            {"id": "other/audio/hyper.opus", "title": "Hyperballad", "duration_s": 322.0, "offset": 0.0, "length": 4.0, "voices": 1, "note": ""},
        ]},
    ],
    "sources": [], "presets": [], "downloads": [], "can_use_youtube": False,
}
SPECS = {"params": {}, "groups": [], "waveforms": [], "waveform_help": {}, "max_duration_s": 120, "tools": {"demucs": False, "basic_pitch": False},
         "drum_params": {"kit": {n: {"low": KIT_SPECS[n].low, "high": KIT_SPECS[n].high, "default": KIT_SPECS[n].default,
                                     "log": KIT_SPECS[n].log, "unit": "", "help": n + " help"} for n in KIT_SPECS}},
         "kit_help": "kit help"}

TRACKS_TEST = """
const realFetch = window.fetch;
window.fetch = async (url, opts) => {
  const path = String(url).split("?")[0];
  const reply = { "/api/specs": %(specs)s, "/api/jobs": [], "/api/library": %(library)s }[path];
  if (reply !== undefined) return new Response(JSON.stringify(reply), { status: 200 });
  return realFetch(url, opts);
};
"""
TRACKS_RUN = """
(async () => {
  await new Promise((r) => setTimeout(r, 300));
  const rows = () => [...document.querySelectorAll("#track-list .track-row")].map((r) => r.querySelector(".track-title").textContent);
  const groups = () => [...document.querySelectorAll("#track-list .track-group")].map((g) => g.textContent);
  const box = document.querySelector("#track-search");
  const type = (text) => { box.value = text; box.dispatchEvent(new Event("input")); };
  const key = (k) => box.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true, cancelable: true }));
  const form = document.querySelector("#upload-form");
  const out = { all: rows(), groups: groups(), count: document.querySelector("#track-count").textContent, kind: document.querySelector("#track").type };
  type("xtal"); out.xtal = rows(); out.xtalCount = document.querySelector("#track-count").textContent;
  type("APHEX cornish"); out.twoWords = rows(); out.twoWordsGroups = groups();
  type("bjork"); out.accent = rows();
  type("hyperbal"); out.part = rows();
  type("zzz"); out.none = rows(); out.noneText = document.querySelector("#track-list").textContent;
  type("");
  [...document.querySelectorAll(".track-row")].find((r) => r.textContent === "Cornish Acid").click();
  out.example = { track: document.querySelector("#track").value, offset: form.elements.offset.value, voices: form.elements.voices.value,
                  dropHidden: document.querySelector("#drop").hidden, button: document.querySelector("#upload-btn").textContent,
                  chosen: document.querySelectorAll(".track-row.chosen").length, note: document.querySelector("#track-note").textContent };
  type("xtal"); key("ArrowDown"); key("Enter");
  out.keyboard = { track: document.querySelector("#track").value, offset: form.elements.offset.value, duration: form.elements.duration.value, stillFiltered: rows() };
  key("Escape"); out.cleared = rows().length;
  [...document.querySelectorAll(".track-row")].find((r) => r.textContent.includes("Upload a file")).click();
  out.upload = { track: document.querySelector("#track").value, dropHidden: document.querySelector("#drop").hidden, button: document.querySelector("#upload-btn").textContent };
  await realFetch("/tracks", { method: "POST", body: JSON.stringify(out) });
})().catch((e) => realFetch("/error", { method: "POST", body: String(e) + " " + e.stack }));
"""


@pytest.fixture(scope="module")
def track_results(tmp_path_factory):
    static = os.path.dirname(KEYBOARD_JS)
    with open(os.path.join(static, "index.html")) as f:
        html = f.read()
    with open(KEYBOARD_JS) as f:
        keyboard = f.read()
    with open(os.path.join(static, "app.js")) as f:
        app = f.read()
    setup = TRACKS_TEST % {"specs": json.dumps(SPECS), "library": json.dumps(LIBRARY)}
    html = html.replace('<script src="keyboard.js"></script>', f"<script>{setup}</script><script>{keyboard}</script>")
    html = html.replace('<script src="app.js"></script>', f"<script>{app}</script><script>{TRACKS_RUN}</script>")
    return json.loads(_run_page_in_chrome(html, 1, tmp_path_factory.mktemp("chrome"))["/tracks"])


@needs_chrome
def test_the_list_shows_every_cached_track_by_dataset(track_results):
    r = track_results
    assert r["all"] == ["Upload a file", "Sparse piano", "Cornish Acid", "Aphex Twin - Cornish Acid", "Aphex Twin - Xtal", "Hyperballad"]
    assert r["groups"] == ["Examples (2)", "Aphex Twin (2)", "Björk search (1)"]
    assert r["count"] == "5 tracks"
    assert r["kind"] == "hidden"


@needs_chrome
def test_searching_narrows_the_list(track_results):
    r = track_results
    assert r["xtal"] == ["Aphex Twin - Xtal"] and r["xtalCount"] == "1 of 5 tracks"
    assert r["twoWords"] == ["Aphex Twin - Cornish Acid"] and r["twoWordsGroups"] == ["Aphex Twin (1 of 2)"]
    assert r["accent"] == ["Hyperballad"]  # by dataset name, ignoring the accent
    assert r["part"] == ["Hyperballad"]
    assert r["none"] == [] and "No tracks match" in r["noneText"]


@needs_chrome
def test_choosing_an_example_fills_in_its_settings(track_results):
    e = track_results["example"]
    assert e["track"] == "aphex_twin/audio/Cornish_Acid.opus" and e["offset"] == "30" and e["voices"] == "2"
    assert e["dropHidden"] and e["button"] == "Fit" and e["chosen"] == 1 and "Acid." in e["note"]


@needs_chrome
def test_the_search_box_works_from_the_keyboard(track_results):
    k = track_results["keyboard"]
    assert k["track"] == "aphex_twin/audio/Xtal.opus" and k["offset"] == "0" and k["stillFiltered"] == ["Aphex Twin - Xtal"]
    assert track_results["cleared"] == 6


@needs_chrome
def test_upload_is_still_there(track_results):
    u = track_results["upload"]
    assert u["track"] == "" and not u["dropHidden"] and u["button"] == "Upload and fit"


# ---- the piano roll's lanes and drum hits, on the real page with the server's answers filled in

def _wav_b64(seconds=1.0):
    import base64
    import io

    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * SR), dtype=np.float32), SR, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()


DRUMS_DESCRIBED = [{"name": name, "gm": gm, "kit": kit, "kits": ["808", "909"], "level": 0.3, "tune": 0.0, "decay": -2.0}
                   for name, gm, kit in (("kick", 36, "808"), ("rim", 37, "909"), ("snare", 38, "909"))]


DRUMKITS = {"default": ["808", "909"], "kits": [
    {"id": "808", "name": "TR-808", "kind": "machine", "drums": ["kick", "snare"], "credit": "", "installed": True},
    {"id": "909", "name": "TR-909", "kind": "machine", "drums": ["kick", "snare"], "credit": "", "installed": True},
    {"id": "linndrum", "name": "LinnDrum (imitation)", "kind": "imitation", "drums": ["kick"], "credit": "", "installed": True},
    {"id": "salamander", "name": "Salamander Drumkit", "kind": "recorded", "drums": [], "credit": "by someone", "installed": False}]}


def _job(job_id, patches, notes, drum_hits):
    return {"id": job_id, "name": job_id, "kind": "upload", "status": "done", "stage": "", "step": 0, "total": 0, "progress": 1.0,
            "loss_curve": [], "error": None, "elapsed_s": 1.0, "created": 1.0, "started": 1.0, "finished": 2.0,
            "config": {"total_steps": 10},
            "result": {"sr": SR, "n_samples": SR, "n_harmonics": 32, "patches": patches, "notes": notes, "roll_notes": notes,
                       "final_loss": 1.0, "tuning_cents": 0.0, "has_noise": True, "drum_hits": drum_hits, "drums": DRUMS_DESCRIBED,
                       "history": [], "reference": None}}


ROLL_TEST = r"""
const wav = Uint8Array.from(atob("%(wav)s"), (c) => c.charCodeAt(0));
const jobs = %(jobs)s;
window.requests = [];
const realFetch = window.fetch;
window.fetch = async (url, opts) => {
  const path = String(url).split("?")[0];
  if (opts && opts.method === "POST") requests.push({ path, body: opts.body });
  if (path.endsWith(".wav") || path.endsWith("/render")) return new Response(wav.slice(), { status: 200, headers: { "X-Spectral-Loss": "1.0" } });
  if (path === "/api/midi") return new Response(new Uint8Array([77, 84, 104, 100]), { status: 200 });
  const job = path.match(/^\/api\/jobs\/(\w+)$/);
  const reply = job ? jobs[job[1]] : { "/api/specs": %(specs)s, "/api/jobs": [], "/api/library": %(library)s, "/api/drumkits": %(drumkits)s }[path];
  if (reply !== undefined) return new Response(JSON.stringify(reply), { status: 200 });
  return realFetch(url, opts);
};
location.hash = "job=both";
"""
ROLL_RUN = """
(async () => {
  const until = async (test) => { for (let i = 0; i < 100 && !test(); i++) await new Promise((r) => setTimeout(r, 100)); };
  await until(() => S.job && S.job.id === "both" && S.drumHits && document.querySelector("#roll").style.height && S.kits);
  const out = {};
  // choosing kits
  const chips = () => [...document.querySelectorAll("#kit-chips .chip")].map((c) => [c.textContent.trim(), c.querySelector("input").checked]);
  const tick = (id, on) => { const cb = document.querySelector(`#kit-chips input[value="${id}"]`); cb.checked = on; cb.dispatchEvent(new Event("change")); };
  const compare = (on) => { const cb = document.querySelector("#compare-kits"); cb.checked = on; cb.dispatchEvent(new Event("change")); };
  out.kits = { chips: chips(), more: document.querySelector("#kit-chips button.link").textContent, value: document.querySelector("#drum-kit").value,
               hint: document.querySelector("#kit-hint").textContent };
  tick("noise", true); out.kits.mixedGeneric = [kitProblem(), document.querySelector("#kit-hint").textContent];
  compare(true); out.kits.comparing = [kitProblem(), document.querySelector("#kit-hint").textContent, document.querySelector("#drum-kit").value];
  tick("808", false); tick("909", false); out.kits.compareOne = kitProblem();
  compare(false); tick("noise", false); out.kits.none = kitProblem();
  tick("808", true); tick("909", true);
  const mode = document.querySelector("#fit-mode");
  mode.value = "notes"; mode.dispatchEvent(new Event("change")); out.kits.notesHidden = document.querySelector("#kit-wrap").hidden; out.kits.notesOk = kitProblem();
  mode.value = "notes_and_drums"; mode.dispatchEvent(new Event("change"));
  // the kits a fit compared
  out.compared = { shown: !document.querySelector("#kit-compare").hidden, note: document.querySelector("#kit-compare-note").textContent,
                   rows: [...document.querySelectorAll("#kit-compare-rows tr")].map((tr) => [tr.className, ...[...tr.children].slice(0, 3).map((td) => td.textContent)]) };
  requests.length = 0;
  document.querySelector("#kit-compare-rows tr:nth-child(2) button").click();
  await until(() => S.play && S.play.which === "kit_1");
  out.compared.playing = S.play && S.play.which;
  stop();
  out.lanes = rollLanes().map((l) => [l.title, l.rows]);
  out.drumRows = rollLanes().find((l) => l.kind === "drums").kinds.map(drumLabel);
  out.metrics = document.querySelector("#metrics").textContent;
  out.synthShown = !document.querySelector(".card.synth").hidden;
  // click the snare hit, where it is drawn
  const c = document.querySelector("#roll"), rect = c.getBoundingClientRect();
  const { lanes, h, dur } = layoutRoll(c);
  const drums = lanes.find((l) => l.kind === "drums");
  const hit = S.drumHits.find((x) => x.kind === 2);
  const y = rect.top + (drums.body + drums.kinds.indexOf(2) * drums.rowH + drums.rowH / 2) / h * rect.height;
  const x = rect.left + (hit.onset + 0.01) / dur * rect.width;
  requests.length = 0;
  c.dispatchEvent(new MouseEvent("click", { clientX: x, clientY: y, bubbles: true }));
  out.muted = S.drumHits.filter((x) => x.muted).map((x) => [x.kind, x.onset]);
  await until(() => requests.some((r) => r.path.endsWith("/render")));
  out.rendered = JSON.parse(requests.find((r) => r.path.endsWith("/render")).body).muted_drum_hits;
  requests.length = 0;
  document.querySelector("#exp-midi").click();
  await until(() => requests.some((r) => r.path === "/api/midi"));
  out.midi = JSON.parse(requests.find((r) => r.path === "/api/midi").body);
  // the drums' sliders: one box per drum that was hit; a louder kick from the other kit, then back
  const boxes = () => [...document.querySelectorAll("#drum-sliders .group")];
  out.drumBoxes = boxes().map((b) => [b.querySelector("h3").textContent, [...b.querySelectorAll(".pname")].map((p) => p.textContent)]);
  out.drumCardShown = !document.querySelector(".card.drums").hidden;
  requests.length = 0;
  const level = boxes()[0].querySelectorAll("input[type=range]")[0];
  level.value = 900; level.dispatchEvent(new Event("input"));
  const kit = boxes()[0].querySelector("select");
  kit.value = "909"; kit.dispatchEvent(new Event("change"));
  await until(() => requests.some((r) => r.path.endsWith("/render") && JSON.parse(r.body).drum_settings[0].kit === "909"));
  out.tweaked = { sent: JSON.parse(requests.filter((r) => r.path.endsWith("/render")).pop().body).drum_settings,
                  title: boxes()[0].querySelector("h3").textContent, lane: rollLanes().find((l) => l.kind === "drums").kinds.map(drumLabel) };
  document.querySelector("#drum-reset-btn").click();
  out.reset = { settings: S.drumSettings, title: boxes()[0].querySelector("h3").textContent, muted: S.drumHits.filter((x) => x.muted).length };
  // a drums-only job: no voices to tweak or play
  await selectJob("drums");
  await until(() => S.job && S.job.id === "drums" && S.drumHits && S.drumHits.length === 1);
  out.drumsOnly = { lanes: rollLanes().map((l) => l.title), synthHidden: document.querySelector(".card.synth").hidden,
                    playHidden: document.querySelector(".card.play").hidden, drumsShown: !document.querySelector(".card.drums").hidden,
                    drumBoxes: [...document.querySelectorAll("#drum-sliders .group h3")].map((h) => h.textContent) };
  await realFetch("/roll", { method: "POST", body: JSON.stringify(out) });
})().catch((e) => realFetch("/error", { method: "POST", body: String(e) + " " + e.stack }));
"""


@pytest.fixture(scope="module")
def roll_results(tmp_path_factory):
    static = os.path.dirname(KEYBOARD_JS)
    with open(os.path.join(static, "index.html")) as f:
        html = f.read()
    with open(KEYBOARD_JS) as f:
        keyboard = f.read()
    with open(os.path.join(static, "app.js")) as f:
        app = f.read()
    patch = {**BASE, "waveform": BASE["waveform"]}
    notes = [{"pitch": 60.0, "onset": 0.1, "duration": 0.3, "velocity": 0.8, "voice": 0},
             {"pitch": 48.0, "onset": 0.5, "duration": 0.3, "velocity": 0.8, "voice": 1}]
    jobs = {"both": _job("both", [patch, patch], notes, [[0, 0.0, 0.9], [2, 0.5, 0.7], [0, 0.75, 0.8]]),
            "drums": _job("drums", [], [], [[0, 0.25, 0.9]])}
    jobs["both"]["result"].update(drum_kits=[{"kit": "808", "loss": 1.2345, "n_hits": 3, "audio": "kit_0"},
                                             {"kit": "linndrum", "loss": 1.5, "n_hits": 5, "audio": "kit_1"}], drum_kits_on="drum stem")
    setup = ROLL_TEST % {"wav": _wav_b64(), "jobs": json.dumps(jobs), "specs": json.dumps(SPECS), "library": json.dumps(LIBRARY),
                         "drumkits": json.dumps(DRUMKITS)}
    html = html.replace('<script src="keyboard.js"></script>', f"<script>{setup}</script><script>{keyboard}</script>")
    html = html.replace('<script src="app.js"></script>', f"<script>{app}</script><script>{ROLL_RUN}</script>")
    return json.loads(_run_page_in_chrome(html, 1, tmp_path_factory.mktemp("chrome"))["/roll"])


@needs_chrome
def test_each_voice_and_the_drums_get_a_lane(roll_results):
    r = roll_results
    assert [title for title, _ in r["lanes"]] == ["Voice 1", "Voice 2", "Drums"]
    assert r["lanes"][2][1] == 2 and r["drumRows"] == ["kick (TR-808)", "snare (TR-909)"]  # only drums that were hit
    assert "3 drum hits" in r["metrics"] and "kick (TR-808) 2" in r["metrics"] and r["synthShown"]


@needs_chrome
def test_clicking_a_drum_hit_mutes_it_in_the_render(roll_results):
    r = roll_results
    assert r["muted"] == [[2, 0.5]]
    assert r["rendered"] == [[2, 0.5]]


@needs_chrome
def test_drums_go_to_midi_by_general_midi_note(roll_results):
    assert roll_results["midi"]["drum_hits"] == [[36, 0.0, 0.9], [36, 0.75, 0.8]]


@needs_chrome
def test_a_drums_only_job_has_only_the_drum_lane(roll_results):
    d = roll_results["drumsOnly"]
    assert d["lanes"] == ["Drums"] and d["synthHidden"] and d["playHidden"]
    assert d["drumsShown"] and d["drumBoxes"] == ["kick (TR-808)"]  # the mute and the sliders are with the drums


# ---- deleting jobs from the list

LIST_TEST = r"""
const wav = Uint8Array.from(atob("%(wav)s"), (c) => c.charCodeAt(0));
const jobs = %(jobs)s;
let listed = Object.values(jobs);
window.requests = [];
window.confirms = [];
window.confirm = (text) => { confirms.push(text); return true; };
const realFetch = window.fetch;
window.fetch = async (url, opts) => {
  const path = String(url).split("?")[0], method = (opts && opts.method) || "GET";
  if (method === "DELETE") requests.push([method, path]);
  if (path.endsWith(".wav") || path.endsWith("/render")) return new Response(wav.slice(), { status: 200, headers: { "X-Spectral-Loss": "1.0" } });
  const job = path.match(/^\/api\/jobs\/(\w+)$/);
  if (job && method === "DELETE") { listed = listed.filter((j) => j.id !== job[1]); return new Response("{}", { status: 200 }); }
  const reply = job ? jobs[job[1]] : { "/api/specs": %(specs)s, "/api/jobs": listed, "/api/library": %(library)s }[path];
  if (reply !== undefined) return new Response(JSON.stringify(reply), { status: 200 });
  return realFetch(url, opts);
};
location.hash = "job=done1";
"""
LIST_RUN = r"""
(async () => {
  const until = async (test) => { for (let i = 0; i < 100 && !test(); i++) await new Promise((r) => setTimeout(r, 100)); };
  const names = () => [...document.querySelectorAll("#jobs .job .jname")].map((x) => x.textContent);
  const cross = (name) => [...document.querySelectorAll("#jobs .job")].find((li) => li.querySelector(".jname").textContent === name).querySelector(".jdel");
  await until(() => S.job && S.job.id === "done1" && names().length === 2);
  const out = { before: names(), titles: [...document.querySelectorAll("#jobs .jdel")].map((b) => [b.textContent, b.title]) };
  cross("run1").click();
  await until(() => names().length === 1);
  out.running = { names: names(), selected: S.jobId, shown: !document.querySelector("#job-view").hidden };
  cross("done1").click();
  await until(() => names().length === 0);
  out.selected = { selected: S.jobId, viewHidden: document.querySelector("#job-view").hidden, hash: location.hash,
                   list: document.querySelector("#jobs").textContent };
  Object.assign(out, { requests, confirms });
  // with Demucs installed, drums-only fits use it unless it was chosen by hand
  const form = document.querySelector("#upload-form").elements, mode = document.querySelector("#fit-mode");
  const pick = (value) => { mode.value = value; mode.dispatchEvent(new Event("change")); return [form.separate.checked, form.voices.disabled, document.querySelector("#kit-wrap").hidden]; };
  out.modes = [pick("drums"), pick("notes"), pick("notes_and_drums")];
  form.separate.click();
  out.chosen = [pick("drums"), pick("notes")];
  await realFetch("/jobs", { method: "POST", body: JSON.stringify(out) });
})().catch((e) => realFetch("/error", { method: "POST", body: String(e) + " " + e.stack }));
"""


@pytest.fixture(scope="module")
def list_results(tmp_path_factory):
    static = os.path.dirname(KEYBOARD_JS)
    with open(os.path.join(static, "index.html")) as f:
        html = f.read()
    with open(KEYBOARD_JS) as f:
        keyboard = f.read()
    with open(os.path.join(static, "app.js")) as f:
        app = f.read()
    patch = {**BASE, "waveform": BASE["waveform"]}
    notes = [{"pitch": 60.0, "onset": 0.1, "duration": 0.3, "velocity": 0.8, "voice": 0}]
    running = {**_job("run1", [], [], []), "status": "running", "progress": 0.4, "finished": None, "result": None}
    jobs = {"run1": running, "done1": _job("done1", [patch], notes, [])}
    specs = {**SPECS, "tools": {"demucs": True, "basic_pitch": False}}
    setup = LIST_TEST % {"wav": _wav_b64(), "jobs": json.dumps(jobs), "specs": json.dumps(specs), "library": json.dumps(LIBRARY)}
    html = html.replace('<script src="keyboard.js"></script>', f"<script>{setup}</script><script>{keyboard}</script>")
    html = html.replace('<script src="app.js"></script>', f"<script>{app}</script><script>{LIST_RUN}</script>")
    return json.loads(_run_page_in_chrome(html, 1, tmp_path_factory.mktemp("chrome"))["/jobs"])


@needs_chrome
def test_each_job_has_a_cross_that_deletes_it(list_results):
    r = list_results
    assert r["before"] == ["run1", "done1"]
    assert r["titles"] == [["×", "Stop and delete this job"], ["×", "Delete this job"]]
    assert r["requests"] == [["DELETE", "/api/jobs/run1"], ["DELETE", "/api/jobs/done1"]]
    assert r["confirms"] == ["Stop and delete “run1”?", "Delete “done1” and its results?"]


@needs_chrome
def test_the_cross_doesnt_open_the_job_and_closes_it_if_open(list_results):
    r = list_results
    assert r["running"] == {"names": ["done1"], "selected": "done1", "shown": True}
    assert r["selected"]["selected"] is None and r["selected"]["viewHidden"] and r["selected"]["hash"] == ""
    assert "No jobs yet" in r["selected"]["list"]


@needs_chrome
def test_drums_only_fits_use_demucs_if_installed_unless_chosen_by_hand(list_results):
    # [Demucs ticked, voices disabled, drum kits hidden] after choosing each fit mode
    assert list_results["modes"] == [[True, True, False], [False, False, True], [False, False, False]]
    assert list_results["chosen"] == [[True, True, False], [True, False, True]]  # ticked by hand: left alone


@needs_chrome
def test_drums_get_sliders_of_their_own(roll_results):
    r = roll_results
    assert r["drumCardShown"]
    assert r["drumBoxes"] == [["kick (TR-808)", ["kit", "level", "tune", "decay"]], ["snare (TR-909)", ["kit", "level", "tune", "decay"]]]
    sent = r["tweaked"]["sent"]
    assert sent[0]["kit"] == "909" and sent[0]["level"] == pytest.approx(0.01 * 200 ** 0.9, rel=1e-3) and sent[2] == {"level": 0.3, "tune": 0.0, "decay": -2.0, "kit": "909"}
    assert r["tweaked"]["title"] == "kick (TR-909)" and r["tweaked"]["lane"] == ["kick (TR-909)", "snare (TR-909)"]
    assert r["reset"]["settings"][0] == {"level": 0.3, "tune": 0.0, "decay": -2.0, "kit": "808"} and r["reset"]["title"] == "kick (TR-808)"
    assert r["reset"]["muted"] == 0  # the snare muted earlier is back too



@needs_chrome
def test_kits_are_chosen_by_ticking_them(roll_results):
    k = roll_results["kits"]
    assert k["chips"] == [["TR-808", True], ["TR-909", True], ["LinnDrum", False], ["Generic sounds", False]]
    assert "Salamander Drumkit" in k["more"] and k["value"] == "808,909" and "whichever" in k["hint"]
    assert "compare them instead" in k["mixedGeneric"][0] and "compare them instead" in k["mixedGeneric"][1]
    assert k["comparing"][0] == "" and "lowest loss" in k["comparing"][1] and k["comparing"][2] == "808,909,noise"
    assert "at least two" in k["compareOne"] and "at least one" in k["none"]
    assert k["notesHidden"] and k["notesOk"] == ""


@needs_chrome
def test_the_kits_compared_are_listed_best_first_and_can_be_heard(roll_results):
    c = roll_results["compared"]
    assert c["shown"] and "drum stem" in c["note"]
    assert c["rows"] == [["used", "TR-808", "1.234", "3"], ["", "LinnDrum", "1.500", "5"]]
    assert c["playing"] == "kit_1"
