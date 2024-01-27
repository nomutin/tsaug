"""Data transformations."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal, TypeVar

import numpy as np
import torch
from einops import pack, reduce, unpack
from torch import Tensor
from torchvision.transforms.functional import to_tensor

from tsaug.utils import softmax


def softmax_transfer(
    data: np.ndarray,
    max_array: np.ndarray,
    min_array: np.ndarray,
    z: int = 7,
    sigma: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Softmax transformation of action data.

    Parameters
    ----------
    data : np.ndarray
        (preferably normalized) action data.
    max_array : np.ndarray
        Max array obtained with get_dim_maxmin().
    min_array : np.ndarray
        Min array obtained with get_dim_maxmin().
    z : int, optional
        Number of nodes constituting one dimension, by default 7.
    sigma : float, optional
        Variance of information represented by node, by default 0.05.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Transformed Array and reference vector.
    """
    ref_vec = np.linspace(min_array, max_array, z, endpoint=True, axis=1)

    batch, length, dim = data.shape
    x = np.empty([batch, length, dim, z])
    for b, t in product(range(batch), range(length)):
        j_vec = -np.square(ref_vec - data[b, t, :][..., np.newaxis]) / sigma
        x[b, t] = softmax(j_vec, axis=-1)
    return x.reshape([batch, length, dim * z]), ref_vec


def inv_sofmax_transfer(
    data: np.ndarray,
    ref_vec: np.ndarray,
) -> np.ndarray:
    """Inverse Softmax transformation."""
    data_batch, data_len, data_dim = data.shape
    ref_dim, num_nodes = ref_vec.shape
    assert data_dim == (ref_dim * num_nodes)

    data = data.reshape([data_batch, data_len, ref_dim, num_nodes])
    y = np.zeros([data_batch, data_len, ref_dim])
    for b, t in product(range(data_batch), range(data_len)):
        y[b, t] = np.sum(data[b, t] * ref_vec, axis=-1)
    return y


T = TypeVar("T", np.ndarray, Tensor)


def get_action_dim_maxmin(data: T) -> tuple[list, list]:
    """Get the max and min values of tensor along the last dim."""
    data, _ = pack([data], "* dim")
    dim_max = reduce(data, "batch dim -> dim", "max")
    dim_min = reduce(data, "batch dim -> dim", "min")
    return dim_max.tolist(), dim_min.tolist()


def copy_arraylike(data: T) -> T:
    """Copy an array-like object, either np.ndarray or torch.Tensor."""
    if isinstance(data, np.ndarray):
        return data.copy()
    return data.detach().clone()


class NormalizeAction:
    """Normalize 3D+ tensor with given max and min values."""

    def __init__(self, max_array: list[int], min_array: list[int]) -> None:
        """Initialize parameters."""
        self.max_array = np.array(max_array)
        self.min_array = np.array(min_array)

    def __call__(self, data: T) -> T:
        """Apply normalization."""
        copy_data = copy_arraylike(data)
        copy_data += -self.min_array
        copy_data *= 1.0 / (self.max_array - self.min_array)
        copy_data *= 2.0
        copy_data += -1.0
        return copy_data


class DenormalizeAction:
    """Denormalize 3D+ tensor with given max and min values."""

    def __init__(self, max_array: list[int], min_array: list[int]) -> None:
        """Initialize parameters."""
        self.max_array = np.array(max_array)
        self.min_array = np.array(min_array)

    def __call__(self, x: T) -> T:
        """Apply denormalization."""
        copy_data = copy_arraylike(x)
        copy_data += 1.0
        copy_data *= 1.0 / 2.0
        copy_data *= self.max_array - self.min_array
        copy_data += self.min_array
        return copy_data


class ToNdarray:
    """Convert Tensor to np.ndarray."""

    def __call__(self, x: Tensor) -> np.ndarray:
        """Convert Tensor to np.ndarray."""
        return x.cpu().detach().numpy()


class ObservationToTensor:
    """Convert observation sequence np.ndarray to Tensor."""

    def __call__(self, data: np.ndarray) -> Tensor:
        """Convert observation sequence np.ndarray to Tensor."""
        array, ps = pack([data], "* h w c")
        tensor = torch.stack([to_tensor(array) for array in array])
        return unpack(tensor, ps, "* h w c")[0]


@dataclass
class FixedActionInfo:
    """Information about (fixed) dimensions of action."""

    dim_idx: int
    constant_value: float | Literal["free"]


def add_fixed_dims_to_action(
    data: np.ndarray,
    robot_joint_info_list: list[FixedActionInfo],
) -> np.ndarray:
    """Add fixed dimensions to action."""
    data_value_list = list(data)
    num_fixed_dims = len(robot_joint_info_list)
    new_data = np.zeros(num_fixed_dims)
    for info in robot_joint_info_list:
        if info.constant_value == "free":
            new_data[info.dim_idx] = data_value_list.pop(0)
        else:
            new_data[info.dim_idx] = info.constant_value

    if data_value_list != []:
        msg = "data_value_list is not empty."
        raise ValueError(msg)

    return new_data
