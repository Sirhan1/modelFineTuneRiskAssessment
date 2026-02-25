import sys

from alignment_risk import cli


def test_cli_no_args_prints_help_instead_of_running_demo(
    monkeypatch,
    capsys,
) -> None:
    called = {"demo": False}

    def _fake_demo(*, output_dir: str = "artifacts", mode: str = "full") -> None:
        _ = output_dir
        _ = mode
        called["demo"] = True

    monkeypatch.setattr(cli, "run_demo", _fake_demo)
    monkeypatch.setattr(sys, "argv", ["alignment-risk"])

    cli.main()
    captured = capsys.readouterr()

    assert "usage:" in captured.out
    assert not called["demo"]
