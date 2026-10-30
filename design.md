# musicfit design notes

How the synthesizer, the fitter, the dataset builder and the web app are put
together, why, what has been measured, and what is known not to work yet.
Usage is in the [README](README.md).

## Synthesizer (`synth.py`)

One voice is a classic subtractive synth, written so every stage is a smooth
function of its parameters:

| stage | parameters (all trainable, sigmoid-bounded to musical ranges, log-scaled where noted in `PARAM_SPECS`) |
|---|---|
| oscillator | mix of sine / saw / square / triangle (softmax weights), built additively from 32 band-limited harmonics |
| filter | 24 dB/oct resonant low-pass; `cutoff`, `resonance`, `filt_env_amount` (octaves) |
| envelopes | amplitude ADSR and filter ADSR (closed form, differentiable in onset and duration) |
| noise | enveloped white noise, `noise_level` |
| effects | `drive` (tanh), feedback delay (`delay_time`, `delay_feedback`, `delay_mix`, exact fractional delay via FFT), noise reverb (`reverb_rt60`, `reverb_mix`), master `gain` |

The filter is applied per harmonic as a magnitude response evaluated at control
rate (every 32 samples) and interpolated, instead of a recursive IIR, so
back-propagation never runs through a sample recursion. That is exact for a
static filter and a good approximation for a slowly moving cutoff; the cost is
no self-oscillation and no filtering of the noise.

Oscillator phase is computed in float64 (2πft gets large) and wrapped to
[0, 2π) before being cast to float32 for `sin`, which halves what autograd
keeps for the biggest tensor, `[notes, harmonics, samples]`.

Speed, forward + backward of one step: a 3 s clip at 16 kHz takes ~0.3–0.4 s
on 48 CPU cores and ~0.02 s on an RTX 3090; a 10 s clip 1–4 s vs. 0.05–0.08 s
with 8.6 GiB peak GPU memory. Memory grows with clip length × (pitches or
notes) × harmonics.

## Note representations (`notes.py`)

* **`PianoRoll`** – a `[pitches, frames]` array of note-on strengths in
  `[0, 1]` (sigmoid of free logits) on a 10 ms grid, with a trainable gate
  length per pitch row. The rendering is linear in the roll, so each row's
  envelopes are one FFT convolution of its onsets with that row's ADSR kernel.
  This is the continuous relaxation that lets gradient descent discover
  *which* notes are played. (It used to have one gate length shared by every
  note of a voice, which forced sustained notes to be rendered as rapid
  re-triggers.)
* **`NoteSequence`** – an explicit list of notes with continuous onset,
  duration, velocity and ±0.5 semitone detune, for refinement after the roll
  has been peak-picked into discrete notes.

Several voices (each a separate patch + roll) can be fitted at once for
polyphonic material with different timbres.

## Fitting (`fit.py`)

```
target audio ──► stage 1: Adam on {piano roll, synth params}
                 loss = MR-STFT(render, target) + λ·(roll mass per frame)          (λ ramps up)
             ──► peak-pick roll → discrete notes
             ──► stage 2: Adam on {note onsets/durations/velocities/detune, synth params}
                 loss = MR-STFT + λ₂·(note velocity sum per frame)
             ──► leave-one-out pruning: drop each note (quietest first) whose removal
                 raises MR-STFT by < prune_tol
             ──► 100 more stage-2 steps at 0.3× learning rate
             ──► Note list, MIDI file, patch JSON, rendered audio
```

The loss is the usual multi-resolution STFT loss (spectral convergence +
log-magnitude L1 over FFT sizes 2048…128), which is phase-insensitive and
gives useful gradients for pitch, timing and timbre at the same time.

Pruning is what fixed most spurious notes. An echo made by the delay effect and
an echo made by an extra note produce the *same audio*, so no loss term can
prefer the delay; only a price per note can. Velocity is a poor proxy for that
price (spurious notes have a median velocity of 0.44 vs. 0.79 for real ones,
with a lot of overlap), but the leave-one-out loss increase
(`removal_deltas()`) separates them well: below 0.01 fall 59 of 113 spurious
notes and 2 of 74 real ones (8 targets, before pruning existed).

`fit()` runs on `FitConfig.device` (`"auto"`: CUDA if available).

## Evaluation on synthetic targets

`synthetic.py` renders a random minor-pentatonic melody (3 s, ~10–13 notes on a
16th grid at 120 bpm, some dyads) with a random patch from
`SynthParams.random`, so notes and patch are known. A note counts as found if
pitch rounds equal and the onset is within 50 ms.

Single runs are not a reliable measure: fits on the GPU are not
deterministic (CUDA's linear-interpolation backward uses atomic adds) and the
optimisation is chaotic, so one target fitted twice with the same settings
gave F1 0.74 and 0.89. Even the mean over 24 targets moves between identical
runs: three runs of the current defaults gave 0.757, 0.715 and 0.764. So
compare settings on many targets, paired by target, and treat differences
below ~0.05 between single 24-target runs as noise.
`scripts/eval_synthetic.py` does this (`--compare` for the paired difference,
`--kinds` for the breakdown of spurious notes below). 24 targets (seeds 1–24),
Standard steps (400 + 300):

| setting | precision | recall | F1 | final loss |
|---|---|---|---|---|
| original fitter (shared gate length, no stage-2 penalty, no pruning) | 0.47 | 0.81 | 0.571 | 0.986 |
| + λ₂ = 3 and leave-one-out pruning at `prune_tol` = 0.01 | 0.60 | 0.81 | 0.670 | 0.947 |
| + a gate length per pitch row (current defaults) | 0.71–0.76 | 0.73–0.78 | 0.715–0.764 | 0.96–0.98 |

The first row is commit 5534654, the second was measured on code between
5534654 and 92ef47a (pruning without per-pitch gates), the third is the
current code over three runs. In single paired runs, pruning gained
+0.099 ± 0.032 (SE) F1 (better on 18 of 24 targets) and the per-pitch gates
another +0.086 ± 0.032 (better on 16); together the gain is well outside the
run-to-run noise, the split between the two less certain.
Predicted notes went 557 → 404 → 291, against 279 real ones. F1 per target
now ranges from 0.32 to 1.0 (median 0.81); the seed-1 demo that used to be
quoted (F1 0.89 then) is one of the better targets, not a typical one. The
Quick preset (150 + 100 steps) does about as well: P 0.70, R 0.83, F1 0.750.

With per-pitch gates, stage 2 no longer improves the note list on average:
over the three runs the notes decoded from the roll score F1 0.759–0.774
(recall ~0.85) and the final ones 0.715–0.764, on average 0.02 lower
(recall drops to 0.73–0.78), whereas with Quick's shorter stage 1 it still adds
+0.09. Pruning at `prune_tol` =
0.005 instead of 0.01 makes no difference (−0.004 ± 0.025), so the lost notes
come from the refinement itself (onsets drifting > 50 ms, or velocities
pushed down by λ₂). Stage 2 still lowers the loss and fits continuous
onsets, durations and detune, so it stays; finding out what it breaks is
the obvious next step.

The remaining 77 spurious notes are onsets 50–150 ms off (24), notes at
harmonic intervals of real ones (13), echoes of real notes one or more delay
times later (12), ±1–2 semitone errors (7), re-triggers inside held notes (7)
and others (14). With pruning but one shared gate length there were 180,
mostly timing (50), echoes (43) and re-triggers inside held notes (39). The fitter still turns
`delay_mix` far below its true value in most fits (to 0.01–0.05 in 7 of the
first 10 targets); at ~0 the delay time gets no gradient, so the delay never
comes back. Failure correlates most with the true drive (r = −0.42) and filter
resonance (r = −0.36; mean F1 0.80 for Q < 2.5 vs. 0.69 for Q ≥ 2.5): a resonant
peak sweeping with the filter envelope gets partly explained by extra notes,
and Q is often under-estimated (e.g. 5.9 → 1.0, with drive 0.24 → 5.1
instead).

Tried and not kept (measured before the per-pitch gates):

* **Echo explain-away move.** Candidate delay times from the onset gaps of
  same-pitch note pairs; for each, drop the notes that would be echoes,
  grid-search delay mix/feedback, accept if the loss rises by less than
  `prune_tol` per dropped note. No gain on 24 targets (F1 0.662 vs. 0.670), and
  it often picked a wrong delay time, because same-pitch repeats on a musical
  grid look like echoes too.
* **Restarts** (4 initialisations, keep the lowest loss + `prune_tol`·notes):
  +0.02 F1 for 4× the compute. The failing targets fail on every restart.
* **More steps.** 1000 + 600 steps (what was the web app's Thorough preset)
  gave F1 0.455: with the warm-up a fixed fraction of the run, stage 1 spends
  longer at full sparsity and drops real notes (recall after stage 1 0.87 →
  0.53). With the stage-1 λ lowered to 0.4 it was still worse, 0.576.
* **Stage-2 L1 alone** (λ₂ = 3, no pruning): F1 0.558 on 8 targets, no gain; it
  helps only together with pruning.

## Real recordings

A 4 s clip of Radiator (SAW II #2, 60 s in, 2 voices) does not fit: stage 1
reaches a loss of 1.23, but the roll is a diffuse cloud of short (~40 ms)
weak onsets (strongest cell 0.17–0.35, decode threshold 0.3), so decoding
keeps 1–24 notes and stage 2 starts from a loss of 3.7–6.9 and ends near 2.9.
Lower thresholds keep hundreds of notes. Per-pitch gate lengths did not
change this. Most likely the sound (a pad, presumably with detuned or
chorused oscillators and heavy reverb) is out of reach of one shared patch
per voice, and the roll ends up painting the spectrogram instead. Real
material probably needs richer voices (unison/detune, a filtered noise or
drum voice) before the note-discovery stage can work on it.

## Dataset builder (`dataset/`)

`build_aphex_twin_dataset.py`, per work:

1. **Catalogue** – MusicBrainz release groups of the artist (by default the
   six major albums + six EPs, matched with `release_key`: "Cheetah EP" =
   "Cheetah", "pt2" = "Part 2"; singles are searched too since MusicBrainz
   types Windowlicker as one; per name the first album, else EP, else single
   is used), tracks of the earliest official release of each. Fallback: the
   curated list in `aphex_twin_catalog.py`, whose durations are MusicBrainz's.
   Works are de-duplicated by a normalised title (case/accents,
   "(2019 Remaster)"-style suffixes, Syro "[120.2]" bpm tags, punctuation).
   Curated search aliases are attached by title or alias, titles first (so the
   RDJ Album track "4" doesn't get SAW II's "#4").
2. **Search** – `ytsearch8:"Aphex Twin <title>"` through yt-dlp (metadata
   only), then each alias, unless an official upload was already found. The
   official SAW II uploads are titled "#n", in two numberings that differ by
   one from Hexagon on, so those tracks carry both. YouTube returns nothing for
   searches containing "Cock", so Cock/Ver10 is also searched as "Ver10".
3. **Choose** – the title must contain the work title (or an alias) as a
   phrase, duration within 8 % (min 6 s) of the catalogue's; official / label
   / "Topic" uploads score +2; covers, remixes, live, slowed, 432/528 Hz
   re-tunes, bass-boosted, reaction, "full album" etc. are rejected unless the
   work's own title contains the word.
4. **Download** – best audio → ffmpeg (Opus kept as is) → `audio/<slug>.opus`.
5. **Manifest** – `manifest.jsonl` (`ok | not_found | duplicate_video |
   download_failed | skipped`); a video id is never downloaded twice; re-runs
   resume; a dry run's picks (`skipped`) are downloaded by the next real run.

Measured (October 2026, home connection, no cookies): 149 works; a full dry
run picks a video for 145, none twice, 132 from the official channel or a
label, all within ~3 s of the MusicBrainz length. The 4 misses are the
Blackbox "omc" mixes, which are only on YouTube as slowed fan edits.

## Web app (`web/`)

Single process. `jobs.py` runs one fit at a time on a worker thread (a fit
saturates the GPU or all CPU cores) and persists each job as `job.json` + WAVs;
jobs that were running when the server stopped are marked failed on restart.
`render.py` re-renders an edited patch / note list without gradients on the
CPU for the sliders (well under a second for a few seconds of audio), so
re-renders compete with a CPU fit but not a GPU one. The frontend is plain JS
with no build step; spectrograms are computed in the browser.

## Limitations / open problems

* Pitch is discrete in stage 1 (one row per semitone) and continuous only
  within ±0.5 st in stage 2; glides and heavy pitch modulation are not
  modelled.
* All notes in a voice share one patch; velocity only scales amplitude and
  filter envelope depth. No unison/detune, no drum or filtered-noise voice.
* Stage 2 loses about 8 % recall that the decoded roll had (see above).
* Echoes are still partly explained as notes (see above); the delay effect
  needs a better way back once it has been turned down.
* High-resonance patches are fitted badly.
* Real recordings do not survive the roll → notes decoding yet.
