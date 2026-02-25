import pytest
import torch

from alignment_risk.utils import (
    batch_size,
    flatten_tensors,
    named_trainable_parameters,
    slice_batch,
)


class _NamedParamsModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor([1.0]))
        self.b = torch.nn.Parameter(torch.tensor([2.0]))
        self.c = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)


def test_batch_size_dict_skips_non_tensor_metadata_values() -> None:
    batch = {
        "meta": "sample-0001",
        "input_ids": torch.randn(3, 4),
        "labels": torch.tensor([0, 1, 0]),
    }
    assert batch_size(batch) == 3


def test_batch_size_sequence_skips_non_tensor_entries() -> None:
    batch = ["meta", torch.randn(5, 2)]
    assert batch_size(batch) == 5


def test_batch_size_skips_scalar_tensor_metadata() -> None:
    batch = {
        "meta_id": torch.tensor(12345),
        "input_ids": torch.randn(4, 6),
        "labels": torch.tensor([0, 1, 0, 1]),
    }
    assert batch_size(batch) == 4


def test_batch_size_sequence_skips_scalar_tensor_entries() -> None:
    batch = [torch.tensor(7), torch.randn(3, 2)]
    assert batch_size(batch) == 3


def test_flatten_tensors_rejects_length_mismatch() -> None:
    params = [
        torch.nn.Parameter(torch.randn(2)),
        torch.nn.Parameter(torch.randn(3)),
    ]
    with pytest.raises(ValueError, match="same length"):
        flatten_tensors([torch.randn(2)], params)


def test_slice_batch_keeps_scalar_tensor_metadata_in_dict() -> None:
    batch = {
        "meta_id": torch.tensor(9),
        "input_ids": torch.randn(4, 3),
        "labels": torch.tensor([0, 1, 0, 1]),
    }
    item = slice_batch(batch, 2)
    assert item["meta_id"].shape == torch.Size([])
    assert item["input_ids"].shape[0] == 1
    assert item["labels"].shape[0] == 1


def test_slice_batch_keeps_scalar_tensor_metadata_in_sequence() -> None:
    batch = [torch.tensor(7), torch.randn(5, 2)]
    item = slice_batch(batch, 1)
    assert item[0].shape == torch.Size([])
    assert item[1].shape[0] == 1


def test_named_trainable_parameters_preserves_explicit_order() -> None:
    model = _NamedParamsModel()
    names, params = named_trainable_parameters(
        model,
        include_names=["b", "a"],
        strict=True,
    )
    assert names == ["b", "a"]
    assert params[0] is model.b
    assert params[1] is model.a


def test_named_trainable_parameters_strict_rejects_missing_names() -> None:
    model = _NamedParamsModel()
    with pytest.raises(ValueError, match="missing names"):
        named_trainable_parameters(
            model,
            include_names=["a", "missing_param"],
            strict=True,
        )


def test_named_trainable_parameters_strict_rejects_non_trainable_names() -> None:
    model = _NamedParamsModel()
    with pytest.raises(ValueError, match="non-trainable names"):
        named_trainable_parameters(
            model,
            include_names=["c"],
            strict=True,
        )
