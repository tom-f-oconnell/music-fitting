"""Multi-resolution STFT loss (spectral convergence + log-magnitude L1)."""

from __future__ import annotations

import torch


class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(self, fft_sizes=(2048, 1024, 512, 256, 128), overlap: int = 4, log_weight: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.fft_sizes = tuple(fft_sizes)
        self.overlap = overlap
        self.log_weight = log_weight
        self.eps = eps
        self._windows: dict[int, torch.Tensor] = {}
        self._target_cache: tuple | None = None

    def _window(self, n: int, device) -> torch.Tensor:
        if n not in self._windows or self._windows[n].device != device:
            self._windows[n] = torch.hann_window(n, device=device)
        return self._windows[n]

    def _mag(self, x: torch.Tensor, n_fft: int) -> torch.Tensor:
        S = torch.stft(x, n_fft, hop_length=n_fft // self.overlap, window=self._window(n_fft, x.device),
                       return_complex=True)
        return S.abs()

    def target_mags(self, target: torch.Tensor) -> list[torch.Tensor]:
        key = (target.data_ptr(), target.shape)
        if self._target_cache is None or self._target_cache[0] != key:
            with torch.no_grad():
                self._target_cache = (key, [self._mag(target, n) for n in self.fft_sizes])
        return self._target_cache[1]

    def _compare(self, P: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        sc = torch.linalg.norm(T - P) / (torch.linalg.norm(T) + self.eps)
        lm = (torch.log(P + self.eps) - torch.log(T + self.eps)).abs().mean()
        return sc + self.log_weight * lm

    def forward(self, pred: torch.Tensor, target: torch.Tensor, spread_cents: float = 0.0) -> torch.Tensor:
        """Loss of pred against target.

        Args:
            pred: predicted waveform.
            target: target waveform.
            spread_cents: if non-zero, compare the spectrum pred would have as a detuned unison
                (see :func:`unison_spectrum`) instead of its own.

        Returns:
            The mean over the FFT sizes.
        """
        loss = pred.new_zeros(())
        for n_fft, T in zip(self.fft_sizes, self.target_mags(target)):
            P = self._mag(pred, n_fft)
            if spread_cents:
                P = unison_spectrum(P, spread_cents)
            loss = loss + self._compare(P, T)
        return loss / len(self.fft_sizes)

    @torch.no_grad()
    def best_spread(self, pred: torch.Tensor, target: torch.Tensor, grid: tuple[float, ...]) -> float:
        """The unison spread from grid that gives the lowest loss for this prediction.

        The prediction's STFTs are computed once, so this costs little more than one
        evaluation of the loss.
        """
        mags = [self._mag(pred, n) for n in self.fft_sizes]
        targets = self.target_mags(target)
        losses = [sum(float(self._compare(unison_spectrum(P, c) if c else P, T)) for P, T in zip(mags, targets))
                  for c in grid]
        return grid[min(range(len(grid)), key=losses.__getitem__)]


def unison_spectrum(mag: torch.Tensor, cents: float) -> torch.Tensor:
    """The expected magnitude spectrogram of three incoherent copies at -cents, 0 and +cents.

    Copies with unrelated phases add in power on average, and detuning a whole sound by a
    factor stretches its spectrum along frequency by that factor.  So the spectrum of a
    unison can be had from one rendered copy, without its beating: a real unison's beats
    depend on phases nobody can recover, and a prediction whose beats fall elsewhere than the
    target's is punished more than one with none (see design.md).

    Args:
        mag: [F, M] magnitude spectrogram, with bins evenly spaced from 0 to Nyquist.
        cents: distance of the outer copies.

    Returns:
        [F, M] magnitude spectrogram.
    """
    F = mag.shape[0]
    power = mag**2
    bins = torch.arange(F, device=mag.device, dtype=mag.dtype)
    out = power
    for ratio in (2.0 ** (cents / 1200), 2.0 ** (-cents / 1200)):
        position = (bins * ratio).clamp(max=F - 1)
        lo = position.floor().long()
        hi = (lo + 1).clamp(max=F - 1)
        frac = (position - lo)[:, None]
        out = out + (1 - frac) * power[lo] + frac * power[hi]
    return torch.sqrt(out / 3 + 1e-14)
