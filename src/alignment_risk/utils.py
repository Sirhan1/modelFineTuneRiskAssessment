from __future__ import annotations

from typing import Any, List, Literal, Sequence, Tuple

import torch

from .types import ParameterSlice


def named_trainable_parameters(
    model: torch.nn.Module,
    include_names: Sequence[str] | None = None,
    *,
    strict: bool = False,
) -> Tuple[List[str], List[torch.nn.Parameter]]:
    all_named = dict(model.named_parameters())
    trainable_named = {
        name: param
        for name, param in all_named.items()
        if param.requires_grad
    }
    if include_names is None:
        return list(trainable_named.keys()), list(trainable_named.values())

    names: List[str] = []
    params: List[torch.nn.Parameter] = []
    seen: set[str] = set()
    missing: List[str] = []
    non_trainable: List[str] = []
    duplicates: List[str] = []

    for name in include_names:
        if name in seen:
            duplicates.append(name)
            continue
        seen.add(name)

        param = all_named.get(name)
        if param is None:
            missing.append(name)
            continue
        if not param.requires_grad:
            non_trainable.append(name)
            continue

        names.append(name)
        params.append(param)

    if strict and (missing or non_trainable or duplicates):
        parts: List[str] = []
        if missing:
            parts.append(f"missing names: {missing}")
        if non_trainable:
            parts.append(f"non-trainable names: {non_trainable}")
        if duplicates:
            parts.append(f"duplicate names: {duplicates}")
        detail = "; ".join(parts)
        raise ValueError(f"Invalid include_names: {detail}.")

    return names, params


def select_parameter_names_for_mode(
    model: torch.nn.Module,
    *,
    mode: Literal["full", "lora"],
    include_names: Sequence[str] | None = None,
    lora_name_markers: Sequence[str] = ("lora_", "lora_A", "lora_B"),
    require_lora_match: bool = True,
) -> List[str]:
    if include_names is None:
        filtered, _ = named_trainable_parameters(model)
    else:
        filtered, _ = named_trainable_parameters(
            model,
            include_names=include_names,
            strict=True,
        )

    if mode == "full":
        return filtered

    # If caller explicitly passed an allowlist, trust it in LoRA mode and skip name heuristics.
    if include_names is not None:
        if not filtered and require_lora_match:
            raise ValueError(
                "PipelineConfig.mode='lora' but no trainable parameters matched fisher.parameter_names. "
                "Ensure your allowlist points to trainable adapter parameters."
            )
        return filtered

    lora_names = [
        name for name in filtered if any(marker.lower() in name.lower() for marker in lora_name_markers)
    ]

    if not lora_names and require_lora_match:
        raise ValueError(
            "PipelineConfig.mode='lora' but no trainable LoRA parameters were found. "
            "Ensure adapters are attached and trainable, or set mode='full'."
        )

    return lora_names


def resolve_device(requested: str | None = "auto") -> torch.device:
    if requested is None or requested == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(requested)


def build_parameter_slices(
    names: Sequence[str], params: Sequence[torch.nn.Parameter]
) -> List[ParameterSlice]:
    slices: List[ParameterSlice] = []
    start = 0
    for name, param in zip(names, params, strict=True):
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
    if len(tensors) != len(reference_params):
        raise ValueError(
            "flatten_tensors requires tensors and reference_params to have the same length: "
            f"got {len(tensors)} and {len(reference_params)}."
        )

    chunks: List[torch.Tensor] = []
    for tensor, param in zip(tensors, reference_params, strict=True):
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
        if batch.ndim == 0:
            raise ValueError("Cannot infer batch size from scalar tensor.")
        return int(batch.shape[0])
    if isinstance(batch, dict):
        for value in batch.values():
            try:
                return batch_size(value)
            except ValueError:
                continue
    if isinstance(batch, (tuple, list)) and batch:
        for value in batch:
            try:
                return batch_size(value)
            except ValueError:
                continue
    raise ValueError("Cannot infer batch size from batch object.")


def slice_batch(batch: Any, index: int) -> Any:
    if torch.is_tensor(batch):
        if batch.ndim == 0:
            return batch
        return batch[index : index + 1]
    if isinstance(batch, dict):
        return {k: slice_batch(v, index) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(slice_batch(x, index) for x in batch)
    if isinstance(batch, list):
        return [slice_batch(x, index) for x in batch]
    return batch
