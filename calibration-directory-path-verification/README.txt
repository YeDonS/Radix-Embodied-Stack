Calibration-directory CLI verification

This evidence is separate from the proposed two-line source change. It checks
path and calibration-file handling against upstream commit
8533e14b9c63f0fd465c6b25bebbce38a63f92e6, using synthetic temporary files.

Run the exact commands in REPRODUCE.txt. That file explains the expected
before/after results and every isolation boundary. verify_calibration_cli.py
executes both complete CLI modules, the actual SOFollower/Robot constructors,
config classes, MotorCalibration, and calibration-file load/save methods.
Hardware methods/factories and model/tensor dependencies are explicit stand-ins.
No devices or model inference were exercised.

The four JSON reports correspond to Python 3.10.20 and Python 3.12.13, each
with draccus 0.8.0. Each report records the verifier SHA256, CLI source SHA256,
base commit and per-scenario outcomes. A passing before report asserts the
known failures; it does not mean the original custom-directory path works.

The patch changes only two assignments in calibrate_so101.py and run_so101_hm.py.
Neither this evidence directory nor its dependencies belong to the PR diff.
