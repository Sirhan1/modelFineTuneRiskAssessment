from alignment_risk.utils import resolve_device


def test_resolve_device_explicit_cpu() -> None:
    device = resolve_device("cpu")
    assert device.type == "cpu"


def test_resolve_device_auto_returns_supported_type() -> None:
    device = resolve_device("auto")
    assert device.type in {"cpu", "mps", "cuda"}


def test_resolve_device_none_matches_auto() -> None:
    a = resolve_device(None)
    b = resolve_device("auto")
    assert a.type == b.type
