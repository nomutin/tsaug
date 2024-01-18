"""Data Augmentations."""

from __future__ import annotations

import random

import torch
import torch.distributions as td
from numpy.random import MT19937, Generator
from torch import Tensor


class AdditiveNoise:
    """
    Add zero-mean Gaussian noise.

    References
    ----------
    * [S4RL](https://arxiv.org/abs/2103.06326v2)
    """

    def __init__(self, std: float = 0.1) -> None:
        """Initialize parameters."""
        self.std = std

    def __call__(self, data: Tensor) -> Tensor:
        """Add noise to data."""
        max_, min_, device = data.max(), data.min(), data.device
        noise = torch.normal(mean=0, std=self.std, size=data.shape)
        return torch.clamp(data + noise.to(device), min=min_, max=max_)


class MultiplicativeNoise:
    """
    Add zero-mean Uniform noise.

    References
    ----------
    * [S4RL](https://arxiv.org/abs/2103.06326v2)
    """

    def __init__(self, low: float = 0.8, high: float = 1.2) -> None:
        """Initialize parameters."""
        self.low = low
        self.high = high

    def __call__(self, data: Tensor) -> Tensor:
        """Add noise to data."""
        epsilon = td.Uniform(low=self.low, high=self.high).sample(data.shape)
        return data * epsilon.to(data.device)


class DynamicsNoise:
    """
    Apply state mix-up.

    References
    ----------
    * [S4RL](https://arxiv.org/abs/2103.06326v2)
    * [mixup](https://arxiv.org/abs/1710.09412)
    """

    def __init__(self, alpha: float = 0.4) -> None:
        """Initialize parameters and random generator."""
        self.alpha = alpha
        self.randgen = Generator(MT19937(42))

    def __call__(self, data: Tensor) -> Tensor:
        """Apply state mix-up."""
        eps_size = [data.shape[0], *[1] * (data.ndim - 1)]
        eps_array = self.randgen.beta(self.alpha, self.alpha, size=eps_size)
        eps_tensor = torch.from_numpy(eps_array).float()
        shift = torch.cat([data[1:].clone(), data[-1:].clone()], dim=0)
        return (eps_tensor * data + (1 - eps_tensor) * shift).to(data.device)


class StateSwitch:
    """
    Apply state switch.

    References
    ----------
    * [S4RL](https://arxiv.org/abs/2103.06326v2)
    """

    def __init__(self, p: float = 0.4) -> None:
        """Initialize parameters."""
        self.p = p

    def process_tensor(self, tensor: Tensor, dim_list: list[int]) -> Tensor:
        """Apply state switch to single timestep tensor."""
        if torch.rand(1).item() < self.p:
            random_dim_list = random.sample(dim_list, len(dim_list))
            return tensor[random_dim_list]
        return tensor

    def __call__(self, data: Tensor) -> Tensor:
        """Apply state switch to time series data."""
        dim_list = list(range(data.shape[-1]))
        tensor_list = [self.process_tensor(i, dim_list) for i in data]
        return torch.stack(tensor_list)


class SequencePadding:
    """Pad sequence with last item."""

    def __init__(self, pad_len: int) -> None:
        """Initialize parameters."""
        self.pad_len = pad_len

    def __call__(self, sequence: Tensor) -> Tensor:
        """Apply padding."""
        pad = torch.stack([sequence[-1]] * self.pad_len, dim=0)
        return torch.cat([sequence, pad], dim=0)


class UniformLengthPadding:
    """Pad sequence to uniform length."""

    def __init__(self, length: int) -> None:
        """Initialize parameters."""
        self.length = length

    def __call__(self, sequence: Tensor) -> Tensor:
        """Apply padding."""
        pad_len = self.length - sequence.shape[0]
        return SequencePadding(pad_len)(sequence)


class RandomWindow:
    """Randomly select a window of sequence."""

    def __init__(self, lower_window_size: int, upper_window_size: int) -> None:
        """Initialize parameters."""
        self.lower = lower_window_size
        self.upper = upper_window_size
        self.randgen1 = Generator(MT19937(42))
        self.randgen2 = Generator(MT19937(42))
        self._update_window_size()

    def _update_window_size(self) -> None:
        """Update window size with `randgen1` ."""
        self.window_size = self.randgen1.integers(self.lower, self.upper)

    def __call__(self, data: Tensor) -> Tensor:
        """Select start idx with `randgen2` and slice data."""
        self._update_window_size()
        seq_len = data.shape[0]
        start_idx = self.randgen2.integers(0, seq_len - self.window_size)
        return data[start_idx : start_idx + self.window_size]
