from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

import torch

from .types import ParameterSlice


def named_trainable_parameters(
    model: torch.nn.Module,
    include_names: Sequence[str] | None = None,
) -> Tuple[List[str], List[torch.nn.Parameter]]:
    allow = set(include_names) if include_names is not None else None
    names: List[str] = []
    params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if allow is not None and name not in allow:
            continue
        names.append(name)
        params.append(param)
    return names, params


def build_parameter_slices(
    names: Sequence[str], params: Sequence[torch.nn.Parameter]
) -> List[ParameterSlice]:
    slices: List[ParameterSlice] = []
    start = 0
    for name, param in zip(names, params):
        width = param.numel()
        end = start + width
        slices.append(ParameterSlice(name=name, start=start, end=end, shape=tuple(param.shape)))
        start = end
    return slices


def flatten_tensors(
    tensors: Sequence[torch.Tensor | None],
    reference_params: Sequence[torch.nn.Parameter],
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for tensor, param in zip(tensors, reference_params):
        if tensor is None:
            chunk = torch.zeros(param.numel(), dtype=param.dtype, device=param.device)
        else:
            chunk = tensor.reshape(-1)
        chunks.append(chunk)
    out = torch.cat(chunks)
    if device is not None:
        out = out.to(device)
    return out


def flatten_parameter_grads(params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    return flatten_tensors([p.grad for p in params], params)


def move_to_device(batch: Any, device: torch.device) -> Any:
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(move_to_device(x, device) for x in batch)
    if isinstance(batch, list):
        return [move_to_device(x, device) for x in batch]
    return batch


def batch_size(batch: Any) -> int:
    if torch.is_tensor(batch):
        return int(batch.shape[0])
    if isinstance(batch, dict):
        for value in batch.values():
            return batch_size(value)
    if isinstance(batch, (tuple, list)) and batch:
        return batch_size(batch[0])
    raise ValueError("Cannot infer batch size from batch object.")


def slice_batch(batch: Any, index: int) -> Any:
    if torch.is_tensor(batch):
        return batch[index : index + 1]
    if isinstance(batch, dict):
        return {k: slice_batch(v, index) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(slice_batch(x, index) for x in batch)
    if isinstance(batch, list):
        return [slice_batch(x, index) for x in batch]
    return batch
