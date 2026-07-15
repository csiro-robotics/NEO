from dataclasses import dataclass, field
from pathlib import Path

import pytest

from neo.utils.commands import run_command
from neo.utils.experiment_io import load_experiment_config, save_experiment_config
from scripts.experiment_config import NeoDataConfig
from scripts.run_assembly import ExperimentConfig as AssemblyConfig
from scripts.run_reconfiguration import ExperimentConfig as ReconfigurationConfig
from scripts.run_removal import ExperimentConfig as RemovalConfig


@dataclass
class NestedConfig:
    enabled: bool = True


@dataclass
class ExampleConfig:
    data_path: Path = Path("data")
    nested: NestedConfig = field(default_factory=NestedConfig)
    _private: str = "not serialized"


def test_experiment_config_round_trip_uses_plain_yaml(tmp_path: Path) -> None:
    source = ExampleConfig(data_path=Path("data with spaces"), nested=NestedConfig(enabled=False))
    config_path = tmp_path / "config.yaml"

    save_experiment_config(source, config_path)
    text = config_path.read_text(encoding="utf8")
    loaded = load_experiment_config(ExampleConfig(), config_path)

    assert "!!python" not in text
    assert "_private" not in text
    assert loaded.data_path == Path("data with spaces")
    assert loaded.nested.enabled is False


def test_unknown_config_keys_fail_with_context(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("nested:\n  unknown: true\n", encoding="utf8")

    with pytest.raises(ValueError, match="nested"):
        load_experiment_config(ExampleConfig(), config_path)


def test_scene_paths_are_mirrored_without_string_replacement() -> None:
    config = NeoDataConfig(
        root_input=Path("datasets/input"),
        outputs_root=Path("artifacts/models"),
        data_type="real",
        action="remove",
        scene_type="single",
        scene_name="data_in_scene_name",
        object_name="object",
    )

    assert config.get_relative_scene_dir() == Path("real/remove/single/data_in_scene_name/object")
    assert config.get_output_dir() == Path("artifacts/models/real/remove/single/data_in_scene_name/object")
    assert config.get_raw_capture_base() == Path(
        "datasets/input/raw_bagfiles/real/remove/single/data_in_scene_name/object"
    )


def test_checked_command_preserves_argument_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}

    def fake_run(args, check):
        observed["args"] = args
        observed["check"] = check

    monkeypatch.setattr("neo.utils.commands.subprocess.run", fake_run)
    run_command(["tool", "--path", Path("directory with spaces")], verbose=False)

    assert observed == {"args": ["tool", "--path", "directory with spaces"], "check": True}


@pytest.mark.parametrize(
    "relative_path, expected_action, config_type",
    [
        ("removal/tomato_soup_can.yaml", "remove", RemovalConfig),
        ("relocation/banana.yaml", "translate_and_rotate", ReconfigurationConfig),
    ],
)
def test_paper_scene_manifests_load(relative_path: str, expected_action: str, config_type) -> None:
    config_path = Path("experiments/paper") / relative_path
    config = load_experiment_config(config_type(), config_path)

    assert config.action == expected_action
    assert config.scene_name is not None
    assert config.object_name is not None


def test_assembly_manifest_runs_all_four_steps() -> None:
    config = load_experiment_config(AssemblyConfig(), Path("experiments/paper/assembly.yaml"))

    assert config.object_names == ["lego_bricks_1", "red_cup", "lego_bricks_2", "golf_ball"]
    assert config.steps_to_run == [0, 1, 2, 3]
