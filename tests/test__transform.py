"""Tests for `dataset.transform`."""

import numpy as np
import pytest
import torch
from einops import repeat
from torch import Tensor

from tsaug.transform import (
    DenormalizeAction,
    NormalizeAction,
    ObservationToTensor,
    ToNdarray,
    copy_arraylike,
    get_action_dim_maxmin,
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


def test_get_action_dim_maxmin(dummy_action: Tensor) -> None:
    """Test `get_action_dim_maxmin`."""
    dim_max, dim_min = get_action_dim_maxmin(dummy_action)
    max_expected = [i + 1 for i in range(dim)]
    min_expected = [-i - 1 for i in range(dim)]
    assert dim_max == max_expected
    assert dim_min == min_expected


class TestCopyArraylike:
    """Tests for `copy_arraylike`."""

    def test_tensor(self, dummy_action: Tensor) -> None:
        """Test case for tensor."""
        tensor_copy = copy_arraylike(dummy_action)
        assert torch.equal(tensor_copy, dummy_action)
        assert tensor_copy is not dummy_action

    def test_numpy(self, dummy_action: Tensor) -> None:
        """Test case for numpy array."""
        array_copy = copy_arraylike(dummy_action.numpy())
        assert torch.equal(torch.from_numpy(array_copy), dummy_action)
        assert array_copy is not dummy_action.numpy()


class TestNormalizeAction:
    """Tests for `normalize_action`."""

    @pytest.fixture(scope="class")
    def normalized_action(self) -> Tensor:
        """Get normalized action tensor [batch, seq_len, dim]."""
        seqs = [torch.linspace(-1, 1, seq_len) for _ in range(dim)]
        sequences = torch.stack(seqs, dim=-1)
        return repeat(sequences, "s d -> b s d", b=batch_size)

    def test_normalize_action(
        self,
        dummy_action: Tensor,
        normalized_action: Tensor,
    ) -> None:
        """Test `normalize_action`."""
        max_, min_ = get_action_dim_maxmin(dummy_action)
        norm = NormalizeAction(max_array=max_, min_array=min_)
        result = norm(dummy_action)
        assert torch.allclose(result.float(), normalized_action.float())


def test__denormalize_action(dummy_action: Tensor) -> None:
    """Test `denormalize_action`."""
    max_, min_ = get_action_dim_maxmin(dummy_action)
    norm = NormalizeAction(max_array=max_, min_array=min_)
    normalized = norm(dummy_action)
    denorm = DenormalizeAction(max_array=max_, min_array=min_)
    denormalized = denorm(normalized)
    assert torch.allclose(denormalized.float(), dummy_action.float())


def test__to_ndarray(dummy_observation: Tensor) -> None:
    """Test `to_ndarray`."""
    ndarray = ToNdarray()(dummy_observation)
    assert isinstance(ndarray, np.ndarray)


class TestObservationToTensor:
    """Tests for `observation_to_tensor`."""

    def test_4d(self) -> None:
        """Test case for 4d observation."""
        size = [batch_size, h, w, c]
        expected_size = [batch_size, c, h, w]
        np_image = np.ones(size, dtype=np.uint8) * 255
        expected = torch.ones(expected_size, dtype=torch.float32)
        output = ObservationToTensor()(np_image)
        assert torch.allclose(output, expected)

    def test_5d(self) -> None:
        """Test case for 5d observation."""
        size = [batch_size, seq_len, h, w, c]
        expected_size = [batch_size, seq_len, c, h, w]
        np_image = np.ones(size, dtype=np.uint8) * 255
        expected = torch.ones(expected_size, dtype=torch.float32)
        output = ObservationToTensor()(np_image)
        assert torch.allclose(output, expected)


class TestAddFixeDimsToAction:
    """Tests for `add_fixed_dims_to_action`."""
