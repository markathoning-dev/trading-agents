from typer.testing import CliRunner
from cli.cgan_cmd import app

runner = CliRunner()


def test_train_bar_mode_flag():
    result = runner.invoke(app, ["train", "--help"])
    assert "--bar-mode" in result.output


def test_generate_bar_mode_flag():
    result = runner.invoke(app, ["generate", "--help"])
    assert "--bar-mode" in result.output


def test_simulate_bar_mode_flag():
    result = runner.invoke(app, ["simulate", "--help"])
    assert "--bar-mode" in result.output