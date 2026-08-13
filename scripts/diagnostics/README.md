# Diagnostics

Author: Dawit Chun

Run `test_scene_dependence.py` on any checkpoint before deploying it, and `latent_health.py` before
trusting a training run. See `../../docs/DIAGNOSTICS.md`.

The scripts under this directory read camera keys and dimensions from the checkpoint's `config.json`
where possible. Those that still take a dataset path expect a LeRobot v3 dataset and accept it as an
argument; none assume a particular task.
