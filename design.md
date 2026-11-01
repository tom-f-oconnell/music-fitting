# musicfit design notes

How the synthesizer, the fitter, the dataset builder and the web app are put
together, why, what has been measured, and what is known not to work yet.
Usage is in the [README](README.md).

## Synthesizer (`synth.py`)

One voice is a classic subtractive synth, written so every stage is a smooth
function of its parameters:

| stage | parameters (all trainable, sigmoid-bounded to musical ranges, log-scaled where noted in `PARAM_SPECS`) |
|---|---|
| oscillator | mix of sine / saw / square / triangle (softmax weights), built additively from `n_harmonics` band-limited harmonics (32; 128 with `--lead`) |
| filter | 24 dB/oct resonant low-pass; `cutoff`, `resonance`, `filt_env_amount` (octaves) |
| envelopes | amplitude ADSR and filter ADSR (closed form, differentiable in onset and duration) |
| pitch | `glide_time`, how fast a note settles when it slides from another (with `glides`); `unison_cents`, three detuned copies (not trained, see "Detuned timbres") |
| noise | enveloped white noise, `noise_level` |
| effects | `drive` (tanh), feedback delay (`delay_time`, `delay_feedback`, `delay_mix`, exact fractional delay via FFT), noise reverb (`reverb_rt60`, `reverb_mix`), master `gain` |

The filter is applied per harmonic as a magnitude response evaluated at control
rate (every 32 samples) and interpolated, instead of a recursive IIR, so
back-propagation never runs through a sample recursion. That is exact for a
static filter and a good approximation for a slowly moving cutoff; the cost is
no self-oscillation and no filtering of the noise.

Oscillator phase is computed in float64 (2πft gets large) and wrapped to
[0, 2π) before being cast to float32 for `sin`, which halves what autograd
keeps for the biggest tensor, `[partials, samples]`. Only partials below
Nyquist are rendered at all, so many harmonics cost little except on low notes.

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
  has been peak-picked into discrete notes. With `glides`, also a slide per
  note: the share of the interval from the previous note that it starts away
  from its own pitch, rendered as a time warp of the oscillators.

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

`fit()` runs on `FitConfig.device` (`"auto"`: CUDA if available). Given
`init_notes` it skips stage 1: the patch and drums are fitted to those notes
for `roll_steps // 4` steps with the notes frozen, then stage 2 starts from them.

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
below ~0.05 between single 24-target runs as noise. The paired standard error
doesn't cover all of this: two runs whose stage 1 settings are identical
differed in stage 1 F1 by +0.033 ± 0.012 and −0.030 ± 0.013. Settings chosen on
seeds 1–24 are therefore re-checked on seeds they weren't chosen on (25–72).
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

**What stage 2 was losing.** With per-pitch gates, stage 2 made the note list
slightly worse than the decoded roll. Following each true note the roll had
found through stage 2 (24 targets, recall 0.855 → 0.758): 208 stayed, 22 were
lost because the onset drifted past the 50 ms tolerance (median error 60 ms), 5
were pruned, 2 went otherwise, and 2 were gained. The lost notes' median velocity
is 0.37, so λ₂ fading them out, which this document used to suspect, is not the
cause. With the linear decay, Adam at the old onset rate of 0.003 can move an
onset up to ~0.5 s over 300 steps, and early in stage 2 the patch is still
adjusting, so onsets chase timbre errors. Candidates on seeds 1–24, where the
fairest measure is the F1 gained in stage 2 (see the noise above):

| fix | F1 | gained in stage 2 | final loss |
|---|---|---|---|
| none | 0.721 | −0.013 | 0.988 |
| **onset rate 0.001** (now the default) | 0.813 | +0.047 | 0.843 |
| onsets bounded to ±25 ms | 0.813 | +0.041 | 0.930 |
| onsets bounded to ±50 ms | 0.755 | −0.007 | 0.952 |
| notes frozen for the first 100 steps | 0.754 | +0.009 | 0.947 |

The lower rate needs no new code and has the lowest loss, and a hard bound
would trap a note whose decoded onset is more than the bound off. On seeds 25–72
it gains +0.046 ± 0.017 F1 (stage 2's own gain −0.025 → +0.050), −0.089 ± 0.019
loss, and +0.036 ± 0.013 on unison targets. Not run: putting pruned notes back
(pruning explains at most 5 of the 29 lost notes) and scoring with a looser
tolerance (changes the score, not the notes).

**The current defaults against the old settings** (onset rate 0.003, no spread
search, sparsity 1.0, which is how to reproduce the "old" column), same code:

| benchmark | old | now | Δ F1 | loss |
|---|---|---|---|---|
| melody, 48 seeds not used for choosing | 0.746 | 0.803 | +0.057 ± 0.018 | 1.023 → 0.835 |
| same, unison ±15 cents | 0.537 | 0.618 | +0.081 ± 0.020 | 1.337 → 1.108 |
| acid line, 24 seeds not used for choosing | 0.479 | 0.489 | +0.009 ± 0.017 | 1.728 → 1.512 |
| BabySlakh piano stems, 20 clips | 0.530 | 0.584 | +0.054 ± 0.033 | 2.078 → 1.721 |
| BabySlakh bass stems, 19 clips | 0.200 | 0.220 | +0.020 ± 0.028 | 1.909 → 1.601 |
| BabySlakh mixes, 20 clips | 0.038 | 0.060 | +0.022 ± 0.008 | 1.701 → 1.644 |

Part of the melody gain is recall bought with more notes: 584 → 671 predicted
against 569 real, and 556 → 839 on unison, whose precision falls from 0.557 to
0.532. The sparsity moved from 1.0 to 0.5 because the command line and the web
app already used 0.5 and the benchmarks should measure what users get; it is
−0.002 ± 0.018 on clean melodies, +0.038 ± 0.026 on piano, +0.027 ± 0.022 on
bass, +0.020 ± 0.009 on mixes, with a lower loss on all four.

The remaining 77 spurious notes (before these changes) are onsets 50–150 ms off (24), notes at
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

Why real tracks didn't fit, and what has been done about it. Clips: 4 s of
aisatsana (40 s in; sparse piano), Cornish Acid (30 s; acid line + drums),
produk 29 (60 s; synths + drums) and Radiator (60 s; pads), 2 voices each.
No ground truth, so the measures are the loss (always the original MR-STFT
loss), the loss right after the roll is decoded into notes, and how
concentrated the roll is.

**What went wrong.** Every clip ended at a loss of 2.0–2.7 (synthetic: ~0.7)
with 1–16 notes, and decoding the roll cost a lot (stage-1 loss 1.0–1.5,
right after decoding 5.1–7.0). On three clips the roll was a diffuse cloud of
~40 ms blips (strongest cell 0.23–0.63; aisatsana, a sparse piano piece,
0.33). Causes found:

* **Tuning.** The tracks are off A440: aisatsana −19 cents, Cliffs −21,
  produk 29 +24 (Radiator ~+46 and Cornish Acid ~0, both uncertain).
  Stage 1's pitch rows were exactly on A440, so off-tuned notes smeared over
  neighbouring rows; fine pitch only became free in stage 2.
* **Drums and other unpitched sound**, which no voice could make, so they
  became spurious or mistimed notes. On Cornish Acid the acid line *was* found
  (strongest cell 0.98), but the drums kept the loss high.
* **Rich, detuned timbres** (unison/chorus, pads).
* Not the cause: the log-magnitude term chasing bins far below the peak.
  84 % of aisatsana's log error is in bins < −60 dB re the peak, but putting
  the log floor at −60 or −80 dB re the peak changed neither the rolls nor the
  final losses.

**A benchmark with ground truth.** `synthetic_target(mismatch=...)` (and
`eval_synthetic.py --mismatch`) adds these to the synthetic targets, one at a
time or together. Cost to the original fitter, F1 paired against the same
24 clean targets (0.764):

| mismatch | F1 | Δ |
|---|---|---|
| drums (kick + hat, −6 dB re the peak) | 0.564 | −0.200 ± 0.047 |
| unison ±15 cents, random phases (with tuning estimation + noise voice, vs. clean 0.715) | 0.495 | −0.220 ± 0.035 |
| unison ±15 cents, phases aligned | 0.656 | −0.108 ± 0.043 |
| tuning +25 cents | 0.643 | −0.120 ± 0.045 |
| noise floor −40 dB | 0.701 | −0.063 ± 0.038 |
| tuning +45 cents | 0.722 | −0.042 ± 0.034 |
| Opus 48 kbit/s | 0.756 | −0.008 ± 0.021 |
| **combined** (tuning 25 + unison 15 + noise −40 + drums −6) | **0.069** | the real-clip failure, reproduced |

**Fixes so far.**

* **Tuning estimation** (`tuning.py`, `FitConfig.tuning_cents = "auto"`):
  the magnitude-weighted circular mean of spectral-peak deviations from the
  semitone grid, within 6 cents on every mismatch; the roll's rows are
  shifted by it. Tuning +25: 0.643 → 0.757 (+0.114 ± 0.042); clean −0.018 ±
  0.019; combined 0.069 → 0.178 (with phase-aligned unison; 0.138 with the
  random-phase unison the benchmark now uses). Below a consistency of 0.2 (white noise
  scores < 0.06) the grid stays at A440, as for Radiator and Cornish Acid.
* **A noise/drum voice** (`drums.py`, `FitConfig.drum_kinds = 3`): drum kinds
  replaying a fixed kernel (band-shaped noise burst + pitch-swept sine,
  peak-normalised so loudness is hit strength × a bounded level) from their
  own onset roll, under 5× the roll sparsity. With the drums in from the first
  step it is a large win on the benchmark: drums 0.564 → 0.750–0.793
  (+0.19–0.23), combined 0.178 → 0.35. On real clips it **takes over**: on
  Cornish Acid the noise voice ends up making 100 % of the output (0 notes;
  one kind became a 23 ms noise burst around 234 Hz fired on every acid note)
  at a *lower* loss (1.66) than notes + drums (1.87) or notes alone (2.22). The
  synth's acid timbre is poor enough (no glides or accents, approximate
  resonance) that coarse band-shaped noise fits the log spectrum better.
  Tighter kernels (tone ≤ 0.3 s ending ≤ 100 Hz, noise ≤ 0.5 s), up to 100×
  the drum sparsity, the peak normalisation and a concave (√) penalty all
  failed to stop it. What works is letting the drums join only after a part
  of stage 1 (`drum_start`), so pitched content is found first: Cornish Acid
  keeps 11–12 notes and produk 29 7–13, at lower losses than without drums.
  But the synthetic drum gain then mostly disappears (+0.042 at 0.5, +0.053 at
  0.25 vs. +0.186 from the start; drums learnt as notes in the first steps
  are not handed back). The default is `drum_start = 0.25`, the safer choice
  for real music; at it, clean targets score 0.736 (−0.028 ± 0.032 vs. no
  fixes), drums 0.617 and combined 0.215 (0.138 with tuning estimation but no
  drums, 0.348 with the drums in from the start). Ground truth on real-sounding
  audio didn't resolve this trade-off (see "BabySlakh").
* A **stationary noise floor** (`noise_floor`, off) helped nowhere (noise −40
  dB: −0.003; combined: none) and is left off.
* **Unison** as amplitude modulation (three copies at ±Δ Hz sum to the
  partial × (1 + 2 cos Δt)) was tried and removed: −0.046 on the unison
  benchmark, +0.027 (n.s.) combined, −0.110 on clean targets. The fitted spread
  gets almost no gradient near zero, and an initial spread adds beating that
  then never goes away.

### BabySlakh: ground truth on real-sounding audio

`scripts/eval_slakh.py` scores against [BabySlakh](https://zenodo.org/records/4603870)
(CC BY 4.0): 20 songs rendered from MIDI with sampled instruments and drums, at
16 kHz. One 4 s clip per song, evenly placed windows, mixes or single instruments.
Data quirks, handled in `slakh.py`: the metadata's `audio_rendered` is false even
for stems that exist, so stems are found by file; bass and a few programs sound an
octave below their MIDI (basic-pitch agrees with written pitch on 2 % of Acoustic
Bass notes and with −12 on 97 %), so they are transposed. Slow attacks make the 50 ms
tolerance harsh: the MIDI-to-audio offset is 6–26 ms for piano, guitar, bass and
leads but 55 ms for pads and 75 ms for strings (F1 0.059 at 50 ms, 0.180 at 100 ms),
so strings are under-scored. The before-and-after table is under "Evaluation on
synthetic targets".

* **On mixes the fitter is not a transcriber.** F1 0.060, 156 notes predicted of
  870. The limit is density, not drums: leaving the drums out of the fit
  (−0.002 ± 0.008), adding them from the first step (0.037 against 0.039) or
  removing them perfectly from the target (0.039) all score the same.
* **So BabySlakh doesn't settle the drum curriculum.** `drum_start` stays at 0.25
  because of the real clips: from the first step the drums still take over Cornish
  Acid (96 % of the output, 1 note) at a *lower* loss than the default (1.607
  against 1.745, 55 % drums, 17 notes).
* **Density.** At the old sparsity of 1.0, piano clips with 37–46 notes get 1–7
  predicted, while sparse clips reach F1 1.0. Going lower gains little: 0.3 scores +0.004 ±
  0.034, with `prune_tol` 0.003 −0.028 ± 0.036, 0.1 with 0.001 −0.061 ± 0.047.
* **Bass octaves.** Among predicted notes the nearest true note is most often an
  octave off: +12 for 55, exact for 39, +19 for 15, and nothing within 50 ms for 93.
  More harmonics don't help (−0.019 ± 0.026).

### Detuned timbres

**Implemented: the expected spectrum, with the spread found by grid search**
(`loss.unison_spectrum`, `FitConfig.spread_grid`). Copies with unrelated phases add
in power on average, and detuning stretches the spectrum, so the magnitude of a
unison can be computed from one rendered copy, without beating; a target's beats
depend on phases nobody can recover, and beats in the wrong place are punished more
than none. The spread's gradient vanishes at 0, so it is searched at a quarter, half
and three quarters of stage 1 and at each refinement. At the end, real copies are
rendered and kept only if they lower the real loss.

**It measures detune but doesn't improve the fit.** On held-out seeds it picks 15
cents on 26 of 48 unison targets and 10–25 on 35, against ≤ 5 cents on 43 of 48 clean
ones. But on the unison targets F1 moves +0.005 ± 0.014 and the loss doesn't move
(−0.002 ± 0.025); on clean ones +0.032 ± 0.013, the same size as the noise above.
Real copies win on only 4–6 of 48 unison targets, so nearly all such fits still sound
like one copy. On the real pad clip (Radiator), real copies at 40 cents, the top of the
grid, were kept in 3 of 6 configurations. It stays on because it is cheap and
harmless, and it reports the spread (`spread_searched` in the history). Nearly all of
the gain on unison in the table above comes from the onset rate and the sparsity, and
detuned targets still score far below clean ones (0.618 against 0.803). **Detune
remains unsolved.**

| approach | F1 on unison | F1 on clean | |
|---|---|---|---|
| expected spectrum + grid search | +0.005 ± 0.014 | +0.032 ± 0.013 | kept, as above |
| smooth both spectra's power in time, stage 1, 40 ms | −0.023 ± 0.019 | −0.018 ± 0.014 | not kept; loss +0.123 ± 0.045 |
| same, 100 ms | +0.023 ± 0.021 | not run | not kept; precision 0.65 → 0.74, but loss +0.106 ± 0.040 |
| same, 40 ms, both stages | −0.028 ± 0.020 | not run | not kept; loss +0.145 ± 0.043 |
| amplitude-modulated unison, spread by gradient | −0.046 | −0.110 | removed earlier: no gradient near 0, and the beats never go away |
| real incoherent copies during training | not run | not run | 3× the memory, and beats fall in the wrong places |
| phases re-randomised each step | not run | not run | a noisy estimate of the same expected spectrum, at 3× the memory |

(Smoothing is against the implemented approach, the rest against no unison.)

### Acid and lead voice

`synthetic.acid_target` is a benchmark the synth can't reproduce exactly: a 303-style
line, 25 % slides and 25 % accents, rendered sample by sample with a saturating 4-pole
ladder at 4× the sample rate. `--lead` (`fit.lead_settings`) improves the old voice in
two ways:

* **128 harmonics instead of 32.** 32 harmonics of a 65 Hz note end at 2.1 kHz, so the
  old voice couldn't make the upper part of a filter sweep at all; 128 reach 7.6 kHz.
* **A slide per note,** which the old voice couldn't express: each note could only
  start at its own pitch.

| 24 held-out targets | F1 | loss | recall, notes reached by a slide | recall, others |
|---|---|---|---|---|
| old settings | 0.479 | 1.728 | 3/85 = 0.04 | 0.67 |
| current defaults | 0.489 | 1.512 | 9/85 = 0.11 | 0.70 |
| `--lead` | 0.478 | 1.431 | 6/85 = 0.07 | 0.67 |
| basic-pitch's notes | 0.799 | 1.702 | 6/85 = 0.07 | 0.96 |
| basic-pitch's notes, `--lead` | 0.811 | 1.588 | 10/85 = 0.12 | 0.96 |

So it fits the sound better (loss −0.080 ± 0.018 against the defaults, lower on 20 of
24) but finds no more notes (−0.011 ± 0.028), and takes about two-thirds longer. It
also places slides where they are: found notes that were reached by a slide start 24 %
of the interval away, others 5–6 %, though that is only 6–10 notes. On the real Cornish
Acid clip three single-run comparisons of the loss gave −0.045, +0.001 and +0.086, so
no consistent gain there. It stays an option, not a default.

What limits F1 isn't the voice: notes reached by a slide have no attack, so stage 1 has
nothing to find, and stage 2 can only refine notes that exist. They are 19 % of these
notes and found about a tenth of the time, by basic-pitch too. A better start helps far
more than a better voice: basic-pitch's notes add +0.33 F1.

A trained accent depth (filter envelope following velocity to a learnt power) was tried
and removed: holding it at 1 was +0.029 ± 0.022 F1 and −0.133 ± 0.088 loss, and told
accents from plain notes no better (AUC 0.67, against 0.64–0.66 trained). Velocity still
scales the filter envelope's depth linearly. A free table of harmonic gains
(DDSP-style) was also removed: bass −0.039 ± 0.027, piano −0.020 ± 0.029.

Not tried: a ladder-shaped magnitude response (cheap, and closer to the real filter's
slope and resonance), a truly differentiable recursive filter (accurate self-oscillation
and saturation, but back-propagation through every sample), no envelope retrigger on a
slide (needed to render a slide faithfully), and a search for notes without attacks,
adding one where the pitch changes and keeping it if the loss falls.

### Pretrained tools (`tools.py`)

Optional, in `requirements-tools.txt`; nothing imports them until they are used. On the
20 BabySlakh mixes:

| method | precision | recall | F1 | loss |
|---|---|---|---|---|
| fit | 0.135 | 0.043 | 0.060 | 1.644 |
| Demucs, then fit | 0.226 | 0.050 | 0.070 | 2.294 |
| basic-pitch alone | 0.382 | 0.479 | 0.412 | |
| Demucs, then basic-pitch | 0.410 | 0.486 | 0.433 | |
| basic-pitch, then fit | 0.380 | 0.476 | 0.410 | 1.414 |
| Demucs, then basic-pitch, then fit | 0.417 | 0.490 | 0.439 | 1.583 |

* **basic-pitch replaces stage 1** (`--basic-pitch`). This is the fix for dense music:
  +0.350 ± 0.032 F1 and −0.230 ± 0.086 loss over the fit alone, better on all 20 clips.
  The two do different jobs: the transcription knows the notes, and the fit adds what a
  transcription doesn't have, a patch and continuous timing that re-create the sound,
  which can be edited and re-rendered. The fit used to throw much of it away (F1 0.280
  with pruning, 0.321 without); now it keeps every note in the pitch range, with no
  penalty on velocity, and matches the transcription's F1 (−0.002 ± 0.003). It also
  serves as a yardstick: on single instruments it scores piano 0.676 (fit 0.584) and
  bass 0.531 (fit 0.220). On real clips, with no ground truth, it lowers the loss on all
  four, most on Radiator (1.739 → 1.256, 3 → 32 notes).
* **Demucs separates first** (`--separate`): notes are fitted to the pitched stems and
  the drum voice to the drum stem, so neither can explain the other's sound, and the
  drums can't take over from the notes (Cornish Acid keeps 22–27). It helps the
  transcription (+0.020 ± 0.006, or +0.029 ± 0.007 with the fit), and does little for
  the fit alone (+0.011 ± 0.008). The price is a higher loss against the mix (+0.169 ±
  0.041 with basic-pitch, +0.650 ± 0.169 without), since the two parts can't make up for
  each other's errors, and each runs one more model. Off unless asked for.

Real clips, current code, 2 voices, single runs (loss, notes, drums' share of the output):

| clip | default | `--lead` | `drum_start` 0 | `--basic-pitch` | `--separate` | both |
|---|---|---|---|---|---|---|
| aisatsana | 1.114, 10, 1 % | 1.088, 11, 1 % | 1.123, 9, 3 % | 1.067, 18, 1 % | 1.403, 10, 0 % | 1.238, 17, 0 % |
| Cornish Acid | 1.745, 17, 55 % | 1.700, 21, 48 % | 1.607, 1, 96 % | 1.573, 20, 68 % | 1.654, 22, 68 % | 1.654, 27, 76 % |
| produk 29 | 1.718, 16, 0 % | 1.716, 20, 4 % | 1.538, 6, 74 % | 1.446, 23, 82 % | 1.742, 12, 32 % | 1.790, 24, 58 % |
| Radiator | 1.739, 3, 1 % | 1.794, 2, 1 % | 1.783, 2, 3 % | 1.256, 32, 0 % | 1.926, 4, 0 % | 1.393, 33, 0 % |

### Next

1. **Voices for transcribed notes.** basic-pitch gives notes without instruments, so all
   of them share one patch. Assigning them to voices is now the largest gap in
   re-creating a dense track.
2. **The drums alongside a transcription.** Without stage 1 there is no `drum_start`: the
   drums join from the first step, and reach 82 % of the output on produk 29.
3. **Bass octaves:** a discrete move after stage 2 that accepts pitch ±12 if the loss falls.
4. **Notes without attacks,** as above; both the fit and basic-pitch miss them.
5. **Detune.** Beating is the likely reason for its extra spurious notes, and the 100 ms
   smoothing's gain in precision is the only lead.

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

**Playing the patch** (`keyboard.js`). The Play box needs a note to start when the key
goes down, which rules out rendering on the server. `LiveSynth` is the synth rebuilt as
a Web Audio graph, at the fit's sample rate, so its harmonics stop where the Python
synth's do: a `PeriodicWave` from the same Fourier series, two cascaded biquads for the
24 dB/oct filter with the filter envelope on `detune` (exactly the synth's exponential
sweep), `setTargetAtTime` for the exponential envelope stages, a `WaveShaper` for the
drive, a `DelayNode` loop and a `ConvolverNode` on a noise impulse response. Other ways
to do it:

* **An AudioWorklet** running a sample-for-sample port would match more closely, but
  browsers only allow worklets over HTTPS or on localhost, and this app is often reached
  over plain HTTP from another machine.
* **Samples pre-rendered by the server** need a new set of renders after every slider
  move, and can't hold a note for as long as the key is down without looping.

`tests/test_keyboard.py` renders ten patches in headless Chrome and compares them with
`render.py`. Levels agree within 2 %; the reverb's varies with its random impulse response
(1.2 in the runs seen, but 0.85–1.31 between noise seeds in Python alone). Above −40 dB the
spectral distance is 0.17–0.68 against 1.4–3.0 for a different patch, apart from the
reverb and noise, whose fine structure is random. Remaining differences: real biquads shift
phase and warp near Nyquist where the Python filter is an ideal magnitude response, and the
fitted drums aren't played. Effects and filter settings reach held notes at once; envelopes
apply from the next note.

The layout is [Ableton Live's](https://www.ableton.com/en/manual/midi-and-key-remote-control/):
keys go by physical position (`KeyboardEvent.code`), so it stays piano-shaped on any
layout. Ableton's manual gives the rows, Z/X and C/V in steps of 20 but not every letter,
so the top end (`;` and `'`) and the velocity ladder 1, 20, …, 120, 127 are from memory of
Live itself. A is MIDI note 60, which Ableton names C3 and this app, like the piano roll,
names C4.

Only boxes that take letters keep the keys. Focus stays behind easily: in a dropdown after
choosing an option, in a number box after typing in it, and, in Safari and in Firefox on a
Mac, even after clicking a button. So number boxes and dropdowns don't stop the keyboard, as
a first version's did, silently. Anything that stops a key is reported in the status line under
the piano, keys light up before any sound is attempted so a sound problem can't look like a
keyboard problem, and a level meter tells a muted computer from a silent synth.

**Cached data** (`dataset/library.py`, `web/downloads.py`). Every subdirectory of
`MUSICFIT_DATA` is a dataset, recognised by its layout (a `manifest.jsonl`, BabySlakh's
`Track*/`, or plain audio files), so data from the command-line builder or copied in by
hand is listed without being re-downloaded, and "downloading" a dataset that's already
there only fetches what is missing. Fits read cached files where they are, rather than
copying them into the job. A run here means fitting one clip, so the list chooses one track;
there is no training over a whole dataset yet, though jobs do queue.

Shipped examples are short BabySlakh clips (CC BY 4.0, `examples/ATTRIBUTION.md`). The
clips the fitter was developed on are copyrighted and this repository is public, so they
aren't committed; they appear among the examples if the data is on disk.

The server has no login and may be exposed to a network, so clients get a narrow API: a
search string, a link whose host is YouTube's, or a Zenodo record number, never an arbitrary
URL (yt-dlp would fetch from a thousand sites, or from inside the network) or a path. Track
ids are resolved to real paths and must stay inside the dataset, as audio files. Archives
are checked against a size limit and unpacked with tar's `data` filter, which refuses
members that land outside the destination. There is deliberately no delete button.

## Limitations / open problems

* Pitch is discrete in stage 1 (one row per semitone) and continuous only
  within ±0.5 st in stage 2. Slides are modelled with `--lead`, but notes reached
  by one are mostly not found; heavy pitch modulation is not modelled.
* All notes in a voice share one patch; velocity only scales amplitude and
  filter envelope depth. Detuned sounds score far below clean ones, and fits of
  them nearly always stay one copy (see "Detuned timbres"). No chorus.
* On dense music the fitter alone finds few notes; basic-pitch's notes fix that
  but all go to one voice (see "Pretrained tools").
* Drum hits are fitted but only exported as a list (`params.json`, the web
  job's `drum_hits`), not as MIDI; the web app has no sliders for them.
* Echoes are still partly explained as notes (see above); the delay effect
  needs a better way back once it has been turned down.
* High-resonance patches are fitted badly.
* Real recordings: see "Real recordings" for what works now and "Next".
