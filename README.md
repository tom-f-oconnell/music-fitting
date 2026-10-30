# musicfit

Fit a small differentiable synthesizer **and** a MIDI-like note sequence to a
clip of electronic music by gradient descent, plus tooling to build a training
set of audio from YouTube (with an Aphex Twin example).

```
musicfit/
  synth.py        differentiable subtractive synth (PyTorch)
  notes.py        soft piano roll, explicit note list, MIDI writer
  loss.py         multi-resolution STFT loss
  fit.py          two-stage joint fitting procedure
  audio_io.py     ffmpeg-based loading, wav saving
  synthetic.py    random melody + random patch targets (ground truth known)
  web/            browser app: upload → background fit → tweak patch live
    app.py          FastAPI routes
    jobs.py         background fit jobs with progress, persisted to disk
    render.py       fast no-gradient re-render + spectral loss
    static/         single-page UI (plain JS, no build step)
  dataset/
    catalog.py            Work records, title normalisation, MusicBrainz client
    aphex_twin_catalog.py curated track list (6 major albums + main EPs)
    youtube.py            yt-dlp search/score/download, manifest, dedup
scripts/
  demo_fit_synthetic.py        render a random melody with a random patch and recover both
  fit_audio.py                 fit a clip of a real recording
  build_aphex_twin_dataset.py  download non-duplicate Aphex Twin works
tests/
```

Install: `pip install -r requirements.txt` (`ffmpeg` must be on `PATH`). Fits run on a CUDA GPU
when one is available (`FitConfig.device = "auto"`; `--device cpu` on the scripts to override), and
CPU PyTorch works too, just 10–35× slower.

Quick start for the web app:

```
pip install -r requirements.txt
uvicorn musicfit.web.app:app --port 8000      # open http://localhost:8000
```

## 1. The synthesizer

One voice is a classic subtractive synth, written so every stage is a smooth
function of its parameters:

| stage | parameters (all trainable, sigmoid-bounded to musical ranges) |
|---|---|
| oscillator | mix of sine / saw / square / triangle, built additively from 32 band-limited harmonics |
| filter | 24 dB/oct resonant low-pass; `cutoff`, `resonance`, `filt_env_amount` (octaves) |
| envelopes | amplitude ADSR and filter ADSR (closed form, differentiable in onset and duration) |
| noise | enveloped white noise, `noise_level` |
| effects | `drive` (tanh), feedback delay (`delay_time`, `delay_feedback`, `delay_mix`, exact fractional delay via FFT), noise reverb (`reverb_rt60`, `reverb_mix`), master `gain` |

The filter is applied per harmonic as a magnitude response evaluated at control
rate (every 32 samples) and interpolated, instead of a recursive IIR, so
back-propagation never runs through a sample recursion. The whole render for a
3 s clip at 16 kHz takes ~0.3–0.4 s forward+backward on a 48-core CPU and
~0.02 s on an RTX 3090 (10 s clip: 1–4 s vs. 0.05–0.08 s, 8.6 GiB peak GPU
memory).

Notes come in two representations (`notes.py`):

* **`PianoRoll`** – a `[pitches, frames]` array of note-on strengths in `[0,1]`
  (sigmoid of free logits) on a 10 ms grid with one shared gate length. The
  rendering is linear in the roll, so all envelopes are one FFT convolution of
  the roll with the ADSR kernel. This is the continuous relaxation that lets
  gradient descent discover *which* notes are played.
* **`NoteSequence`** – an explicit list of notes with continuous onset,
  duration, velocity and ±0.5 semitone detune, for refinement after the roll
  has been peak-picked into discrete notes.

Several voices (each a separate patch + roll) can be fitted at once for
polyphonic material with different timbres.

## 2. Fitting (`fit.py`)

```
target audio ──► stage 1: Adam on {piano roll, synth params}
                 loss = MR-STFT(render, target) + λ·(roll mass per frame)   (λ ramps up)
             ──► peak-pick roll → discrete notes
             ──► stage 2: Adam on {note onsets/durations/velocities/detune, synth params}
                 loss = MR-STFT
             ──► prune quiet notes → Note list, MIDI file, patch JSON, rendered audio
```

The loss is the usual multi-resolution STFT loss (spectral convergence +
log-magnitude L1 over FFT sizes 2048…128), which is phase-insensitive and
gives useful gradients for pitch, timing and timbre at the same time.

```
python scripts/demo_fit_synthetic.py --out demo_out      # synthetic sanity check
python scripts/fit_audio.py clip.opus --offset 10 --duration 4 --voices 2 --out fits/clip
```

The demo prints note precision/recall against the ground-truth melody and the
recovered vs. true synth parameters (see "Status" below for numbers).

## 3. Web app (`musicfit/web`)

```
uvicorn musicfit.web.app:app --port 8000                          # http://localhost:8000
MUSICFIT_JOBS=/data/musicfit_jobs uvicorn musicfit.web.app:app    # where jobs are stored (default ./web_jobs)
```

What it does:

* **Fit a clip.** Upload any file ffmpeg can decode, choose the start and length
  (up to 15 s), number of voices, and a quality preset (Quick 150+100 steps,
  Standard 400+300, Thorough 1000+600; advanced: pitch range, sparsity, seed).
  The clip is peak-normalised and fitted on a background worker thread, one job
  at a time (a fit already uses every core); further jobs queue. Or click
  **Run demo fit** to fit the synthetic random-melody target, where the
  ground-truth notes and patch are shown for comparison.
* **Progress.** Stage (piano roll / note refinement), step, elapsed and
  estimated remaining time, a live loss curve, and Cancel.
* **Result.** Log-frequency spectrograms of the target and of the current
  render (same dB scale, click to seek), a piano roll of the fitted notes
  coloured by voice (hover for pitch/onset/length/velocity; ground-truth notes
  outlined for the demo), note precision/recall for the demo, and playback of
  target / fitted / tweaked with A/B switching at the same position, loop, and
  space to play/stop.
* **Tweak.** A slider for every patch parameter (log-scaled where the
  parameter is, with editable values that accept `ms`/`k` suffixes) plus the
  four oscillator-mix weights, grouped by stage and per voice. Each slider
  marks the fitted value (and the true value in the demo); double-click resets
  it. Any change re-renders server-side with the fitted notes in roughly
  0.1–0.3 s for a few-second clip (no fitting; just a forward pass) and
  restarts playback at the same position, so you hear the change at once;
  the spectral loss of the tweaked render against the target is shown next to
  the fitted one. Clicking a note mutes it, and voices can be muted.
  Tweaks are kept per job in the browser's local storage.
* **Export.** Patch JSON (all voices, `{"format": "musicfit-patch", "voices": [...]}`,
  re-importable with *Import patch*), MIDI and a notes JSON of the unmuted notes, and the
  tweaked render as WAV.

API (all JSON unless noted): `GET /api/specs`, `GET /api/jobs`,
`POST /api/jobs` (multipart: `file`, `offset`, `duration`, `voices`,
`roll_steps`, `note_steps`, `sparsity`, `pitch_min`, `pitch_max`, `seed`),
`POST /api/jobs/demo`, `GET /api/jobs/{id}` (status, progress, loss curve,
result with patches and notes), `POST /api/jobs/{id}/cancel`,
`DELETE /api/jobs/{id}`, `GET /api/jobs/{id}/audio/{target|fit}.wav`,
`POST /api/jobs/{id}/render` (`{patches, notes, muted_voices}` → WAV, loss in
the `X-Spectral-Loss` header), `POST /api/midi` (`{notes}` → .mid).

There is no authentication: run it locally or behind something that adds it.

## 4. Dataset from YouTube (`musicfit/dataset`)

`build_aphex_twin_dataset.py` does, per work:

1. **Catalogue** – track list from MusicBrainz (all official albums/EPs of the
   artist, restricted by default to the six major albums + main EPs), with a
   curated hardcoded list as fallback (`aphex_twin_catalog.py`: SAW 85–92,
   SAW II, …I Care Because You Do, Richard D. James Album, Drukqs, Syro; EPs
   Come to Daddy, Windowlicker, CCAI pt2, Cheetah, Collapse, Blackbox Life
   Recorder; durations taken from MusicBrainz). Release names are matched
   loosely ("Cheetah EP" = "Cheetah", "pt2" = "Part 2"), and singles are
   searched too, since MusicBrainz types Windowlicker as one. Works are
   de-duplicated by a normalised title (case/accents, "(2019 Remaster)"-style
   suffixes, Syro "[120.2]" bpm tags, punctuation), so re-issues collapse into
   one work.
2. **Search** – `ytsearch8:"Aphex Twin <title>"` through yt-dlp (metadata
   only), then under aliases (the SAW II "#n" names the official uploads use,
   etc.) unless an official upload was already found.
3. **Choose** – candidates are scored: the title must contain the work title
   as a phrase, duration within 8 % (min 6 s) of the catalogue duration,
   official / label / "Topic" uploads preferred, and covers, remixes, live,
   slowed, 432 Hz re-tunes, reaction, "full album" videos etc. rejected
   (unless the work's own title contains that word, e.g. "Come to Daddy
   (Pappy Mix)").
4. **Download** – best audio stream → ffmpeg → `audio/<Artist>_<Title>.opus`
   (the 48 kHz Opus stream is kept as is).
5. **Manifest** – `manifest.jsonl`, one line per work with status
   (`ok | not_found | duplicate_video | download_failed | skipped`), chosen
   video, uploader, durations, file and the scored candidate list. A video id
   already in the manifest is never downloaded twice, and re-running resumes
   (a real run after `--dry-run` downloads the dry run's picks).

```
python scripts/build_aphex_twin_dataset.py --out data/aphex_twin                 # MusicBrainz → download
python scripts/build_aphex_twin_dataset.py --out data/aphex_twin --dry-run        # pick videos only
python scripts/build_aphex_twin_dataset.py --out data/aphex_twin --source curated --no-eps
python scripts/build_aphex_twin_dataset.py --out data/aphex_twin --cookies cookies.txt
python scripts/build_aphex_twin_dataset.py --out data/aphex_twin --all-releases    # every official release
```

`--cookies` (a Netscape `cookies.txt` exported from a logged-in browser) is
the standard workaround when YouTube answers "Sign in to confirm you're not a
bot", typically from data-centre IPs (not needed from a home connection).
Searches containing "Cock" return nothing on YouTube, so Cock/Ver10 is
searched as "Ver10". Downloading copyrighted music is subject to
YouTube's terms and local law; this code is for personal research use.

## Tests

```
python -m pytest            # unit tests; network tests are deselected
python -m pytest -m network # also hit MusicBrainz
```

`tests/test_dataset.py` exercises title normalisation, de-duplication,
candidate scoring and the whole download/manifest loop with a mocked YouTube
client. `tests/test_synth.py` checks envelopes, waveform spectra, roll vs.
note-list rendering agreement, gradient flow to every parameter, roll decoding,
the MIDI writer and a tiny end-to-end fit.

## Status (what has actually been run)

* **Synthetic demo** (`scripts/demo_fit_synthetic.py --seed 1`, 3 s clip, 13
  ground-truth notes incl. dyads, random patch, CPU only, ~7 min): the fit
  recovers 12 of the 13 notes with 14 predicted (precision 0.86, recall 0.92,
  F1 0.89, onsets within 50 ms) and gets the timbre parameters close
  (cutoff 935 → 919 Hz, resonance 1.80 → 1.63, amp attack 23 → 22 ms, amp
  release 181 → 185 ms, noise 0.026 → 0.029, gain 0.50 → 0.50, waveform mix
  72/25 % sine/saw → 60/27 %). The missed note is the upper note of a final
  dyad that got merged into a longer preceding note of the same pitch. Delay
  time/mix and reverb RT60 are only loosely recovered: the spectral loss is
  weakly sensitive to echoes, and in fact the two spurious notes sit exactly
  one delay time (163 ms) after real notes, i.e. the fitter explained the
  echoes as extra quiet notes instead of as the delay effect. A loss term
  on the long-window autocorrelation, or fitting the effects first, would
  address that. The sparsity weight matters a lot
  (λ = 0.2 / 0.5 / 1.0 gave F1 0.39 / 0.57 / 0.92 on a shorter run); the
  default is λ = 1.0 with a linear warm-up.
* **Web app**: run end to end with uvicorn + headless Chromium. A synthetic
  demo job on the Quick preset (150 + 100 steps, 3 s clip, 4-core CPU) took
  3.7 min and found all 13 ground-truth notes but with 20 spurious quiet ones
  (precision 0.39, recall 1.0); use Standard (the CLI defaults above, F1 0.89)
  for a cleaner note list. Slider changes re-render in well under a second,
  the spectral loss readout updates, and upload → progress → cancel works
  from the browser, including at phone width.
* **Unit tests**: 19 pass (`python -m pytest`), including the full
  search → choose → download → manifest loop against a mocked YouTube client
  and web API tests (demo job, upload job, re-render, mute, MIDI, bad input).
* **Dataset, live** (October 2026, home connection, no cookies): MusicBrainz
  gives 149 distinct works for the six albums + six EPs. A full `--dry-run`
  picks a video for 145 of them, no video twice, 132 from the official
  channel or label (all 24 SAW II tracks as the official "#n" uploads, every
  pick within ~3 s of the MusicBrainz length). The 4 misses are the
  Blackbox "omc" mixes, which are only on YouTube as slowed fan edits. Real
  downloads (yt-dlp → `.opus`) were run for the first two works only.
* **Not yet done**: fitting a real recording end to end. `fit_audio.py` is
  wired up (ffmpeg decode → fit → wav/MIDI/JSON) but real tracks will need
  more voices, a drum/noise voice, and longer runs than the demo; treat the
  current fitter as a starting point for that.

## Formulation notes / limitations

* Pitch is discrete in stage 1 (one row per semitone) and continuous only
  within ±0.5 st in stage 2; glides and heavy pitch modulation are not
  modelled.
* All notes in a voice share one patch; velocity only scales amplitude and
  filter envelope depth.
* The filter is a magnitude-only approximation applied per harmonic; there is
  no self-oscillation, and the noise component is not filtered.
* GPU memory grows with clip length × (pitches or notes) × harmonics: about
  2 GiB for a 3 s clip and 8.6 GiB for 10 s with 49 pitches and 32
  harmonics. Reduce `n_harmonics` or the pitch range for longer clips.
