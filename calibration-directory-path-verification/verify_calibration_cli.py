#!/usr/bin/env python3
"""Verify complete CLI modules with real LeRobot configuration and Robot code.

Requires Python >=3.10 and draccus==0.8.0. Run with:
  python verify_calibration_cli.py REPO --expect buggy|fixed --output result.json

No AST extraction or rewriting is used. Camera/robot configs, MotorCalibration,
Robot and SOFollower are imported from the checkout. Both real constructors
run, as do Robot's calibration-file load/save methods. Hardware methods,
factories and tensor/policy imports are explicit stand-ins. Runtime uses zero
control steps and a stub policy loader. This verifies path and file handling,
not inference, real calibration or motion. All robot-related file operations
are confined to temporary synthetic workspaces.
"""
from __future__ import annotations

import argparse
import dataclasses
from enum import Enum
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types


def install_module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    sys.modules[name] = module
    return module


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def forbidden(*args, **kwargs):
    raise AssertionError("Unexpected tensor, policy, or hardware operation in path verification")


def dependencies(repo, workspace):
    source = repo / "third_party/lerobot/src/lerobot"
    for suffix in ("", ".cameras", ".cameras.opencv", ".cameras.realsense", ".robots",
                   ".robots.so_follower", ".utils", ".datasets", ".policies", ".motors"):
        install_module("lerobot" + suffix, __path__=[str(source / suffix.lstrip(".").replace(".", "/"))])

    load_module("lerobot.utils.errors", source / "utils/errors.py")
    load_module("lerobot.utils.decorators", source / "utils/decorators.py")
    install_module("serial", Serial=forbidden)
    install_module("deepdiff", DeepDiff=forbidden)
    install_module("tqdm", tqdm=forbidden)
    install_module("lerobot.utils.utils", enter_pressed=forbidden, move_cursor_up=forbidden)
    motors = load_module("lerobot.motors.motors_bus", source / "motors/motors_bus.py")
    for name in ("Motor", "MotorCalibration", "MotorNormMode"):
        setattr(sys.modules["lerobot.motors"], name, getattr(motors, name))
    install_module("lerobot.processor", RobotAction=dict, RobotObservation=dict)
    install_module("lerobot.utils.constants", HF_LEROBOT_CALIBRATION=workspace / "hf-calibration",
                   ROBOTS="robots", OBS_STR="observation")
    cameras = load_module("lerobot.cameras.configs", source / "cameras/configs.py")
    sys.modules["lerobot.cameras"].CameraConfig = cameras.CameraConfig
    load_module("lerobot.cameras.opencv.configuration_opencv", source / "cameras/opencv/configuration_opencv.py")
    load_module("lerobot.cameras.realsense.configuration_realsense", source / "cameras/realsense/configuration_realsense.py")
    load_module("lerobot.robots.config", source / "robots/config.py")
    load_module("lerobot.robots.so_follower.config_so_follower", source / "robots/so_follower/config_so_follower.py")
    robot_module = load_module("lerobot.robots.robot", source / "robots/robot.py")
    instances = []

    class BusDouble:
        def __init__(self, *, port, motors, calibration):
            self.port, self.motors, self.calibration = port, motors, calibration
            self.is_connected = False

        def __getattr__(self, name):
            raise AssertionError(f"Unexpected motor bus operation: {name}")

    class OperatingMode(Enum):
        POSITION = 0

    install_module("lerobot.motors.feetech", FeetechMotorsBus=BusDouble, OperatingMode=OperatingMode)
    install_module("lerobot.cameras.utils", make_cameras_from_configs=lambda configs: {
        name: types.SimpleNamespace(config=config, is_connected=False) for name, config in configs.items()
    })
    install_module("lerobot.robots.utils", ensure_safe_goal_position=forbidden)
    follower = load_module("lerobot.robots.so_follower.so_follower", source / "robots/so_follower/so_follower.py")
    follower_package = load_module("lerobot.robots.so_follower", source / "robots/so_follower/__init__.py")
    Follower = follower_package.SO101Follower
    assert Follower is follower.SOFollower and Follower.name == "so_follower"
    original_initialize = Follower.__init__
    real_constructor = str(Path(inspect.getsourcefile(original_initialize)).relative_to(repo))

    def initialize(self, configuration):
        original_initialize(self, configuration)
        self.events = []
        instances.append(self)

    def connect(self, **kwargs):
        self.events.append("connect")
        self.bus.is_connected = True
        for camera in self.cameras.values():
            camera.is_connected = True

    def calibrate(self):
        self.events.append("calibrate")

    def disconnect(self):
        self.events.append("disconnect")
        self.bus.is_connected = False
        for camera in self.cameras.values():
            camera.is_connected = False

    Follower.__init__ = initialize
    Follower.connect, Follower.calibrate, Follower.disconnect = connect, calibrate, disconnect

    install_module("numpy")
    install_module("torch")
    install_module("lerobot.datasets.utils", build_dataset_frame=forbidden)
    install_module("lerobot.policies.factory", make_pre_post_processors=forbidden)
    install_module("lerobot.policies.utils", make_robot_action=forbidden)
    install_module("lerobot.utils.control_utils", predict_action=forbidden)
    install_module("modeling_pi05_hm", ARTIFACT_DIR=workspace / "artifacts", HMPI05Policy=object)
    return robot_module, Follower, motors.MotorCalibration, instances, real_constructor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    parser.add_argument("--expect", choices=("buggy", "fixed"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    original_argv, original_cwd = sys.argv[:], Path.cwd()
    results = []
    fixture = {
        name: {"id": i + 1, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
        for i, name in enumerate(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"))
    }
    with tempfile.TemporaryDirectory(prefix="radix-cli-validation-") as temporary:
        root = Path(temporary)
        robot_module, Follower, MotorCalibration, instances, real_constructor = dependencies(repo, root)
        for entrypoint in ("calibrate_so101.py", "run_so101_hm.py"):
            cli = load_module("review_" + Path(entrypoint).stem, repo / entrypoint)
            if entrypoint == "run_so101_hm.py":
                cli.load_policy = lambda path: (None, None, None)
            for case in ("omitted", "empty", "relative", "absolute", "spaces", "unicode", "existing", "current",
                         "existing_calibration", "new_roundtrip", "custom_robot_id", "path_is_file",
                         "parent_is_file", "malformed_calibration"):
                workspace = root / Path(entrypoint).stem / case
                workspace.mkdir(parents=True)
                os.chdir(workspace)
                fallback = workspace / "hf-calibration"
                robot_module.HF_LEROBOT_CALIBRATION = fallback
                values = {
                    "omitted": None, "empty": "", "relative": "calibration/custom",
                    "absolute": str(workspace / "absolute" / "calibration"),
                    "spaces": "calibration with spaces", "unicode": "校准数据/机械臂",
                    "existing": "existing", "current": ".",
                    "existing_calibration": "saved", "new_roundtrip": "roundtrip",
                    "custom_robot_id": "id-specific", "path_is_file": "not-a-directory",
                    "parent_is_file": "parent-file/nested", "malformed_calibration": "malformed",
                }
                value = values[case]
                if case == "existing":
                    (workspace / value).mkdir()
                robot_id = "validation_arm" if case == "custom_robot_id" else "so101_follower"
                expected = Path(value) if value else fallback / "robots" / Follower.name
                calibration_file = expected / f"{robot_id}.json"
                original_calibration_bytes = None
                if case in {"existing_calibration", "custom_robot_id"}:
                    expected.mkdir(parents=True)
                    calibration_file.write_text(json.dumps(fixture, indent=2) + "\n")
                    original_calibration_bytes = calibration_file.read_bytes()
                    if case == "custom_robot_id":
                        (expected / "so101_follower.json").write_text("{ this other robot file must not be opened")
                elif case == "path_is_file":
                    Path(value).write_text("synthetic regular file")
                elif case == "parent_is_file":
                    Path("parent-file").write_text("synthetic regular file")
                elif case == "malformed_calibration":
                    expected.mkdir()
                    calibration_file.write_text("{ malformed synthetic JSON")
                error_expected = {
                    "path_is_file": FileExistsError,
                    "parent_is_file": NotADirectoryError,
                    "malformed_calibration": json.JSONDecodeError,
                }.get(case)
                command = [entrypoint]
                if entrypoint == "run_so101_hm.py":
                    model = workspace / "model"
                    model.mkdir()
                    (model / "config.json").write_text(json.dumps({
                        "type": "hm_pi05", "output_features": {"action": {"shape": [6]}}
                    }))
                    command += ["--model-path", str(model), "--task", "synthetic validation",
                                "--top-camera", "0", "--side-camera", "1", "--steps", "0", "--no-calibrate"]
                if value is not None:
                    command += ["--calibration-dir", value]
                if case == "custom_robot_id":
                    command += ["--robot-id", robot_id]
                sys.argv = command
                before = len(instances)
                should_fail = args.expect == "buggy" and bool(value)
                try:
                    cli.main()
                except Exception as exc:
                    if should_fail:
                        assert isinstance(exc, AttributeError) and str(exc) == "'str' object has no attribute 'mkdir'", exc
                        outcome = "expected upstream mkdir AttributeError"
                    else:
                        assert error_expected is not None and isinstance(exc, error_expected), (case, type(exc), str(exc))
                        outcome = f"expected invalid-input {type(exc).__name__}"
                    assert len(instances) == before
                else:
                    assert not should_fail, f"Expected upstream failure: {entrypoint} {case}"
                    assert error_expected is None, f"Invalid input accepted: {entrypoint} {case}"
                    assert len(instances) == before + 1
                    robot = instances[-1]
                    assert isinstance(robot.calibration_dir, Path), type(robot.calibration_dir)
                    assert robot.calibration_dir == expected, (robot.calibration_dir, expected)
                    assert robot.calibration_fpath == calibration_file
                    assert robot.calibration_dir.is_dir()
                    expected_events = ["connect", "calibrate", "disconnect"] if entrypoint.startswith("calibrate") else ["connect", "disconnect"]
                    assert robot.events == expected_events, robot.events
                    assert robot.config.id == robot_id and robot.config.port == "/dev/ttyACM0"
                    assert robot.robot_type == "so_follower"
                    assert not robot.is_connected
                    assert robot.bus.calibration is robot.calibration
                    assert set(robot.bus.motors) == set(fixture)
                    assert set(robot.cameras) == set(robot.config.cameras)
                    if original_calibration_bytes is not None:
                        assert all(isinstance(item, MotorCalibration) for item in robot.calibration.values())
                        assert {key: dataclasses.asdict(item) for key, item in robot.calibration.items()} == fixture
                        assert calibration_file.read_bytes() == original_calibration_bytes
                    if case == "new_roundtrip":
                        assert not calibration_file.exists() and robot.calibration == {}
                        robot.calibration = {name: MotorCalibration(**fields) for name, fields in fixture.items()}
                        robot._save_calibration()
                        assert json.loads(calibration_file.read_text()) == fixture
                        reloaded = Follower(robot.config)
                        assert {key: dataclasses.asdict(item) for key, item in reloaded.calibration.items()} == fixture
                        assert reloaded.events == [] and not reloaded.is_connected
                    outcome = "expected successful path initialization and cleanup"
                results.append({"entrypoint": entrypoint, "case": case, "outcome": outcome,
                                "existing_file_preserved": original_calibration_bytes is not None and not should_fail,
                                "calibration_roundtrip_checked": case == "new_roundtrip" and not should_fail})
    os.chdir(original_cwd)
    sys.argv = original_argv
    report = {
        "harness_revision": "2: actual SOFollower constructor and calibration file lifecycle",
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "python": sys.version, "draccus": importlib.metadata.version("draccus"), "expect": args.expect,
        "source_sha256": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
                          for name in ("calibrate_so101.py", "run_so101_hm.py")},
        "checks_passed": len(results), "checks": results,
        "outcome_counts": {kind: sum(kind in row["outcome"] for row in results)
                           for kind in ("successful", "upstream mkdir", "invalid-input")},
        "actual_follower_name": Follower.name, "actual_constructor_source": real_constructor,
        "real_code": ["complete CLI modules", "RobotConfig", "SO101FollowerConfig", "SOFollower.__init__",
                      "Robot.__init__", "Robot._load_calibration", "Robot._save_calibration", "MotorCalibration",
                      "OpenCVCameraConfig", "RealSenseCameraConfig", "draccus 0.8.0"],
        "substituted": ["SO101 hardware methods", "motor bus/camera factories", "unused heavy motor helpers",
                        "tensor/policy imports", "runtime policy loading"],
        "limitations": ["No M50 SDK or model execution", "No serial or camera I/O", "Runtime uses zero control steps"],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{args.expect}: {len(results)} checks passed; evidence: {args.output}")


if __name__ == "__main__":
    main()
