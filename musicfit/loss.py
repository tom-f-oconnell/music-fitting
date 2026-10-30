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

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = pred.new_zeros(())
        for n_fft, T in zip(self.fft_sizes, self.target_mags(target)):
            P = self._mag(pred, n_fft)
            sc = torch.linalg.norm(T - P) / (torch.linalg.norm(T) + self.eps)
            lm = (torch.log(P + self.eps) - torch.log(T + self.eps)).abs().mean()
            loss = loss + sc + self.log_weight * lm
        return loss / len(self.fft_sizes)
