# Physics from Video

Code for **Physics from Video: Identifiability of Time-Invariant Second-Order ODEs under Minimal Trajectory Conditions**.

This repository estimates the parameters of a time-invariant second-order ODE directly from video. The main experiments are organized as:

- `synthetic/`: paper experiments on controlled single-video and multi-video settings.
- `robustness/`: appendix robustness experiments for perturbed/noisy visual environments added during rebuttal.

The estimated dynamics use

```text
z''(t) + gamma1 z'(t) + gamma0 z(t) = 0
```

where an encoder maps frames to a scalar latent trajectory `z(t)`, and `gamma0`, `gamma1` are learned by minimizing the finite-difference ODE residual with a non-collapse regularizer.

## Installation

```bash
cd PhysicsFromVideo
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CUDA is used automatically when PyTorch detects a GPU. To force CPU, pass `--device cpu` to the runners below.

## Repository Layout

```text
PhysicsFromVideo/
  synthetic/
    run_experiment.py        # unified runner for main paper experiments
    pendulum_single.py       # original single-video training script
    pendulum_multi.py        # original multi-video training script
    intensity_single.py
    intensity_multi.py
    scale_single.py
    scale_multi.py
    syn_gen/                 # synthetic video generators
  robustness/
    run_robustness.py        # unified runner for appendix noisy-env experiments
    main_ablation.py
    data_gen/                # noisy environment generators and setting registry
  scripts/
    run_synthetic_paper.sh
    run_robustness_paper.sh
```

Generated videos and results are intentionally ignored by git. Recreate them with the commands below.

## Main Paper Experiments

Run from `synthetic/`:

```bash
cd synthetic
```

Generate all synthetic datasets:

```bash
python run_experiment.py generate --dataset all
```

Run the single-video experiments:

```bash
python run_experiment.py single --dataset all --seeds 0 1 2 3 4
```

Run the multi-video experiments:

```bash
python run_experiment.py multi --dataset all --seeds 0 1 2 3 4
```

Run the full synthetic paper sweep:

```bash
python run_experiment.py paper --dataset all --seeds 0 1 2 3 4
```

For a quick smoke test, reduce optimization steps:

```bash
python run_experiment.py paper --dataset pendulum --seeds 0 --steps 10 --device cpu
```

Outputs are written to:

```text
synthetic/data/
synthetic/results/
```

The default synthetic experiment matrix is:

| Dataset | Setting | Single-video input | Multi-video inputs |
| --- | --- | --- | --- |
| `pendulum` | undamped | `theta0=90`, `dtheta0=0`, `0.5pi`, `6 fps` | `dtheta0 in {0, 100, -100}` |
| `intensity` | critical | `z0=1`, `dz0=0`, `2 s`, `10 fps` | `dz0 in {0, 200, -200}` |
| `scale` | undamped | `z0=1`, `dz0=0`, `2 s`, `60 fps` | `dz0 in {0, 10, -10}` |

## Robustness Appendix Experiments

Run from `robustness/`:

```bash
cd robustness
```

Generate all perturbed-environment videos:

```bash
python run_robustness.py generate --setting all
```

Train on all noisy settings:

```bash
python run_robustness.py train --setting all --seeds 42
```

Run generation plus training:

```bash
python run_robustness.py paper --setting all --seeds 42
```

Quick smoke test:

```bash
python run_robustness.py paper --setting noise_background_underdamped --levels 1 --steps 10 --device cpu
```

The noisy environment settings are defined in `robustness/data_gen/setting_registry.py`:

- `noise_background_underdamped`
- `moving_clutter_underdamped`
- `camera_jitter_underdamped`
- `brightness_drift_underdamped`
- `brightness3d_underdamped`
- `brightness3d_lightdom_underdamped`
- `occlusion_underdamped`

Outputs are written to:

```text
robustness/data_gen/ablation_data/
robustness/results/
```

## Notes

- The default training budget is `30000` steps, matching the original scripts. Use `--steps` for faster debugging.
- The generated videos are small but numerous; they are ignored by git to keep the repository lightweight.
- Summary CSV files contain the learned `gamma0` and `gamma1` values for each run.
- The original per-experiment scripts are kept for transparency, while `run_experiment.py` and `run_robustness.py` are the recommended public entry points.
