"""Gradients of a long render, a few notes at a time.

A voice's dry audio is a sum over its notes (or the pitch rows of its piano roll), and the
effects and the loss only ever see that sum. So by the chain rule

    d loss / d theta = sum over batches of <d loss / d mix, d batch / d theta>

which can be computed in two passes: render every batch without gradients and backpropagate the
loss only as far as the mix, then render each batch again with gradients, weighted by the result,
and free it before the next. Only one batch's graph is ever in memory (see design.md).
"""

from __future__ import annotations

from typing import Callable

import torch

# Renders one batch, returning its dry audio and the sample of the clip where that audio starts.
Batch = Callable[[], tuple[torch.Tensor, int]]


@torch.no_grad()
def render_mix(batches: list[Batch], n_samples: int, device) -> torch.Tensor:
    """The sum of the batches, as [n_samples] audio without gradients."""
    mix = torch.zeros(n_samples, device=device)
    for render in batches:
        y, start = render()
        mix[start:start + y.numel()] += y
    return mix


def backward_in_batches(
    voices: list[list[Batch]],
    n_samples: int,
    device,
    loss_of_mixes: Callable[[list[torch.Tensor]], torch.Tensor],
) -> torch.Tensor:
    """Accumulate the gradients of a loss of the voices' mixes in the parameters, as loss.backward() would.

    Args:
        voices: each voice's batches.
        n_samples: length of the mixes.
        device: where to keep the mixes.
        loss_of_mixes: the loss given one [n_samples] dry mix per voice. Anything else it depends on,
            such as the effects' parameters, gets its gradients as usual.

    Returns:
        The loss.
    """
    if len(voices) == 1 and len(voices[0]) == 1:  # no need for two passes
        y, start = voices[0][0]()
        loss = loss_of_mixes([torch.nn.functional.pad(y, (start, n_samples - start - y.numel()))])
        loss.backward()
        return loss.detach()
    mixes = [render_mix(batches, n_samples, device).requires_grad_(True) for batches in voices]
    loss = loss_of_mixes(mixes)
    loss.backward()
    for mix, batches in zip(mixes, voices):
        if mix.grad is None:
            continue
        for render in batches:
            y, start = render()
            if y.requires_grad:  # a note that has moved off the end of the clip renders nothing
                (y * mix.grad[start:start + y.numel()]).sum().backward()
    return loss.detach()
