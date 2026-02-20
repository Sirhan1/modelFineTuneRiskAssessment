import pytest
import torch

from alignment_risk.utils import select_parameter_names_for_mode


class _ModelWithLora(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(8, 8)
        self.lora_A = torch.nn.Parameter(torch.randn(2, 8))
        self.lora_B = torch.nn.Parameter(torch.randn(8, 2))


class _ModelNoLora(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base = torch.nn.Linear(8, 8)


def test_full_mode_includes_all_trainable_parameters() -> None:
    model = _ModelWithLora()
    names = select_parameter_names_for_mode(model, mode="full")
    assert "base.weight" in names
    assert "base.bias" in names
    assert "lora_A" in names
    assert "lora_B" in names


def test_lora_mode_includes_only_lora_parameters() -> None:
    model = _ModelWithLora()
    names = select_parameter_names_for_mode(model, mode="lora")
    assert names == ["lora_A", "lora_B"]


def test_lora_mode_errors_when_no_lora_parameters_found() -> None:
    model = _ModelNoLora()
    with pytest.raises(ValueError, match="mode='lora'"):
        select_parameter_names_for_mode(model, mode="lora", require_lora_match=True)


def test_lora_mode_can_be_non_strict() -> None:
    model = _ModelNoLora()
    names = select_parameter_names_for_mode(model, mode="lora", require_lora_match=False)
    assert names == []
