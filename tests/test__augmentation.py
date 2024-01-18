"""Tests for `dataset.transform`."""

import pytest
import torch
from einops import repeat
from torch import Tensor

from tsaug.augmentation import (
    AdditiveNoise,
    DynamicsNoise,
    MultiplicativeNoise,
    SequencePadding,
    UniformLengthPadding,
)

batch_size, seq_len, dim = 4, 32, 8
c, h, w = 3, 16, 24


@pytest.fixture()
def dummy_action() -> Tensor:
    """Create dummy action tensor [batch, seq_len, dim]."""
    seqs = [torch.linspace(-i, i, seq_len) for i in range(1, dim + 1)]
    sequences = torch.stack(seqs, dim=-1)
    return repeat(sequences, "s d -> b s d", b=batch_size)


@pytest.fixture()
def dummy_observation() -> Tensor:
    """Create dummy observation tensor [batch, seq_len, c, h, w]."""
    size = [batch_size, seq_len, c, h, w]
    return torch.randint(low=254, high=255, size=size)


class TestAdditiveNoise:
    """Tests for `AdditiveNoise`."""

    def test__no_noise(self, dummy_action: Tensor) -> None:
        """Test case for no noise."""
        noise = AdditiveNoise(std=0.0)
        result = noise(dummy_action)
        assert torch.equal(result, dummy_action)

    def test__noise(self, dummy_action: Tensor) -> None:
        """Test case for noise."""
        noise = AdditiveNoise(std=1.0)
        result = noise(dummy_action)
        assert not torch.equal(result, dummy_action)


class TestMultiplicativeNoise:
    """Tests for `MultiplicativeNoise`."""

    def test__no_noise(self, dummy_action: Tensor) -> None:
        """Test case for no noise."""
        noise = MultiplicativeNoise(low=1.0, high=1.0 + 1e-6)
        result = noise(dummy_action)
        assert torch.allclose(result, dummy_action)

    def test__noise(self, dummy_action: Tensor) -> None:
        """Test case for noise."""
        noise = MultiplicativeNoise(low=0.8, high=0.8 + 1e-6)
        result = noise(dummy_action)
        assert torch.allclose(result.div(0.8), dummy_action)


class TestDynamicsNoise:
    """Tests for `DynamicsNoise`."""

    def test__no_noise(self, dummy_action: Tensor) -> None:
        """Test case for no noise."""
        noise = DynamicsNoise(alpha=1e-6)
        result = noise(dummy_action)
        assert torch.allclose(result, dummy_action)

    def test__noise(self, dummy_action: Tensor) -> None:
        """
        Test case for noise.

        Note:
        ----
        This test is not yet rigorous.
        """
        noise = DynamicsNoise(alpha=0.4)
        result = noise(dummy_action)
        assert not torch.equal(result, dummy_action)


class TestStateSwitch:
    """Tests for `StateSwitch`."""


class TestSequencePadding:
    """Tests for `SequencePadding`."""

    def test__action(self, dummy_action: Tensor) -> None:
        """Test case for action."""
        pad_len = 30
        dummy_sequence = dummy_action[0]
        padding = SequencePadding(pad_len=pad_len)
        result = padding(dummy_sequence)
        assert result.shape == torch.Size([seq_len + pad_len, dim])
        assert torch.equal(result[-1], dummy_sequence[-1])
        assert torch.equal(result[-pad_len], dummy_sequence[-1])
        assert not torch.equal(result[-pad_len - 2], dummy_sequence[-1])


class TestUniformLengthPadding:
    """Tests for `UniformLengthPadding`."""

    def test__48(self, dummy_action: Tensor) -> None:
        """Test case for `length=48`."""
        length = 48
        dummy_sequence = dummy_action[0]
        pad = UniformLengthPadding(length=length)
        result = pad(dummy_sequence)
        assert result.shape == torch.Size([length, dim])
        assert torch.equal(result[-1], dummy_sequence[-1])
        assert not torch.equal(result[30], dummy_sequence[-1])

    def test__64(self, dummy_action: Tensor) -> None:
        """Test case for `length=64`."""
        length = 64
        dummy_sequence = dummy_action[0]
        pad = UniformLengthPadding(length=length)
        result = pad(dummy_sequence)
        assert result.shape == torch.Size([length, dim])
        assert torch.equal(result[-1], dummy_sequence[-1])
        assert not torch.equal(result[30], dummy_sequence[-1])
