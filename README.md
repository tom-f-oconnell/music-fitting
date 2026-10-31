# musicfit

Fits a small differentiable synthesizer *and* a MIDI-like note sequence to a
short clip of electronic music by gradient descent, so you get back the notes
and a synth patch that (approximately) re-create it. Comes with a web app for
fitting a clip and tweaking the fitted patch by ear, and a tool that builds a
training set of audio from YouTube (Aphex Twin as the example).

## Using the web app

You need Python 3.10 or newer and [ffmpeg](https://ffmpeg.org/download.html).
In a terminal, in this folder:

```
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/uvicorn musicfit.web.app:app --port 8000
```

Then open http://localhost:8000.

* **Fit a clip:** choose an audio file (anything ffmpeg can read), where the
  clip starts and how long it is (up to 15 s), how many voices (synth
  patches) to fit, and a preset (Standard, or Quick, which leaves slightly
  more spurious notes). Or click **Run demo fit** to fit a random synthetic melody
  whose true notes and patch are shown for comparison.
* **Listen and compare:** spectrograms of the original and the fit, a piano
  roll of the fitted notes, and playback of original / fit / your tweaked
  version, switching at the same position (space plays and stops).
* **Tweak:** every patch parameter has a slider, which marks the fitted
  value; double-click resets it. Changes re-render within a second. Click a
  note to mute it. Drums and other unpitched sound are fitted separately and
  stay in every re-render unless you tick *Mute drums & noise*.
* **Mono lead or acid line:** under Advanced, this also fits slides between
  notes and more harmonics, and takes about two-thirds longer.
* **Export:** patch JSON (re-importable), MIDI, notes JSON, and the WAV.

It works best on short, sparse synth lines. On its own it finds few of the
notes in busy music, so for anything dense install the two optional helpers
below and tick *Start from an automatic transcription* under Advanced (see
[design.md](design.md#real-recordings) for measurements). A fit takes about a
minute on an NVIDIA GPU, which is used automatically, and several minutes on
a CPU. There is no login, so only run the app on your own machine.

**Optional helpers for busy music** (Advanced shows their checkboxes once they
are installed; Demucs downloads its model the first time it runs):

```
venv/bin/pip install -r requirements-tools.txt
venv/bin/pip install --no-deps basic-pitch
```

## Command line and configuration

```
python scripts/fit_audio.py clip.opus --offset 10 --duration 4 --voices 2 --out fits/clip
python scripts/demo_fit_synthetic.py --out demo_out      # synthetic target, prints note P/R and patch
python scripts/eval_synthetic.py --seeds 1-24 --out eval_out/base.json                # 24 targets, mean P/R/F1
python scripts/eval_synthetic.py --set prune_tol=0.005 --compare eval_out/base.json   # paired comparison
python scripts/eval_slakh.py --source mix --out eval_out/slakh_mix.json               # scored against real notes
```

`fit_audio.py` writes `target.wav`, `fit.wav`, `noise.wav` (the fitted drums
alone), `notes.mid`, `notes.json` and `params.json` (patches, the estimated
tuning, drum hits). Options: `--sr`, `--pitch-min/--pitch-max` (MIDI),
`--roll-steps`, `--note-steps`, `--sparsity` (stage-1 penalty: raise for fewer
spurious notes, lower if notes go missing), `--drum-kinds` (unpitched sounds to
fit, default 3; 0 for none), `--noise-floor`, `--seed`, `--device` (`auto`,
`cpu`, `cuda`, ...), `--lead` (for a mono lead or acid line: fit slides between
notes, with 128 harmonics). With the optional helpers: `--basic-pitch` starts
from a transcription instead of finding notes itself, and keeps all its notes,
and `--separate` splits off the drums with Demucs and fits notes and drums each
to their own stem. The recording's tuning offset from A440 is estimated
automatically.
`eval_synthetic.py` fits many synthetic targets (in parallel on a GPU, ~5 min
for 24) with any `FitConfig` overrides (`--set field=value`), optionally with
things real recordings have and the synth can't make (`--mismatch
tuning=25`, `unison`, `noise`, `drums`, `codec`) or on a 303-style line
(`--target acid`); use it rather than single fits to judge a change (see
[design.md](design.md#evaluation-on-synthetic-targets)).
`eval_slakh.py` scores against the true notes of
[BabySlakh](https://zenodo.org/records/4603870) (20 songs rendered from MIDI, 883
MB; its docstring has the download), on mixes or one instrument (`--source`),
with `--method fit`, `basic-pitch` or `fit+basic-pitch`, and `--separate`.
From Python, `musicfit.fit.fit(audio_tensor, FitConfig(...))`; `FitConfig`
documents every knob.

**Web app:** `MUSICFIT_JOBS=/path` sets where jobs are stored (default
`./web_jobs`). HTTP API (JSON unless noted): `GET /api/specs`, `GET /api/jobs`,
`POST /api/jobs` (multipart: `file`, `offset`, `duration`, `voices`,
`roll_steps`, `note_steps`, `sparsity`, `pitch_min`, `pitch_max`, `seed`, `lead`,
`separate`, `basic_pitch`; the last two only if `tools` in `/api/specs` says they are installed),
`POST /api/jobs/demo`, `GET /api/jobs/{id}`, `POST /api/jobs/{id}/cancel`,
`DELETE /api/jobs/{id}`, `GET /api/jobs/{id}/audio/{target|fit}.wav`,
`POST /api/jobs/{id}/render` (`{patches, notes, muted_voices}` → WAV, loss in
the `X-Spectral-Loss` header), `POST /api/midi` (`{notes}` → .mid).

**Dataset from YouTube:**

```
python scripts/build_aphex_twin_dataset.py --out data/aphex_twin --dry-run   # pick videos, download nothing
python scripts/build_aphex_twin_dataset.py --out data/aphex_twin             # download (resumes; keeps dry-run picks)
python scripts/build_aphex_twin_dataset.py --out data/aphex_twin --source curated --no-eps --limit 10
python scripts/build_aphex_twin_dataset.py --out data/aphex_twin --all-releases
```

It takes the track list from MusicBrainz (curated fallback built in), finds
the official upload (or the best other one) of each track on YouTube,
skips covers/remixes/re-tunes and duplicates, and writes `audio/*.opus`, a
`catalog.json` and a `manifest.jsonl` with the status and chosen video of
every track. If YouTube answers "Sign in to confirm you're not a bot"
(typical from data-centre IPs), pass `--cookies cookies.txt` exported from a
logged-in browser. Downloading copyrighted music is subject to YouTube's
terms and local law; this is for personal research use.

## Development

```
musicfit/
  synth.py        differentiable subtractive synth (PyTorch)
  drums.py        unpitched voice: drum kinds from an onset roll, noise floor
  notes.py        soft piano roll, explicit note list, MIDI writer
  loss.py         multi-resolution STFT loss, expected spectrum of a detuned unison
  tuning.py       the recording's offset from A440
  fit.py          two-stage joint fitting procedure
  tools.py        optional Demucs and basic-pitch, and fitting with them
  audio_io.py     ffmpeg-based loading, wav saving
  synthetic.py    synthetic targets with known notes: random melodies, 303-style lines
  slakh.py        BabySlakh clips with their true notes
  web/            FastAPI app (app.py), background jobs (jobs.py), fast re-render (render.py), static/ UI
  dataset/        MusicBrainz catalogue + title matching (catalog.py), curated Aphex Twin list, yt-dlp pipeline (youtube.py)
scripts/          fit_audio.py, demo_fit_synthetic.py, eval_synthetic.py, eval_slakh.py, build_aphex_twin_dataset.py
tests/
```

[design.md](design.md) describes how the synth and the fitter work, the
evaluation on synthetic targets and BabySlakh (and why single runs are
misleading), what was tried, the dataset matching rules, and the open problems.

```
venv/bin/python -m pytest             # unit tests (network tests deselected)
venv/bin/python -m pytest -m network  # also query MusicBrainz
```
