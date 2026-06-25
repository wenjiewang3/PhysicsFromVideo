#!/usr/bin/env python3
"""Runner for appendix robustness experiments under perturbed environments."""

from __future__ import annotations

import argparse
import csv
import os
import random
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import main_ablation
from data_gen.setting_registry import get_available_setting_cases, get_setting_levels


ROOT = Path(__file__).resolve().parent
DATA_GEN = ROOT / "data_gen"

GENERATOR_SCRIPTS = {
    "noise_background_underdamped": "noise_background_gen.py",
    "moving_clutter_underdamped": "moving_clutter_gen.py",
    "camera_jitter_underdamped": "camera_jitter_gen.py",
    "brightness_drift_underdamped": "brightness_drift_gen.py",
    "brightness3d_underdamped": "brightness3d_gen.py",
    "brightness3d_lightdom_underdamped": "brightness3d_lightdom_gen.py",
    "occlusion_underdamped": "occlusion_gen.py",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def selected_settings(name: str) -> list[str]:
    if name == "all":
        return [s for s in get_available_setting_cases() if s in GENERATOR_SCRIPTS]
    return [name]


def generate_setting(setting: str) -> None:
    script = GENERATOR_SCRIPTS[setting]
    print(f"[INFO] Generating robustness videos for {setting}")
    subprocess.run([sys.executable, script], cwd=DATA_GEN, check=True)


def video_path(setting: str, level: int, duration_pi: float, fps: int, theta0: int, dz: int) -> Path:
    filename = f"{duration_pi:.1f}pi_{fps}fps_degree{theta0}_degs{dz}_level{level}.mp4"
    return DATA_GEN / "ablation_data" / setting / filename


def train_setting(
    setting: str,
    seed: int,
    steps: int,
    device: str,
    levels: list[int] | None,
) -> None:
    fps = 20
    duration_pi = 2.5
    theta0 = 60
    dz = 0
    level_list = levels if levels else get_setting_levels(setting)
    result_base = ROOT / "results" / "pendulum_single"
    save_dir = result_base / f"{setting}_new" / f"seed{seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    summary_path = result_base / f"{setting}_summary_seed{seed}.csv"
    new_file = not summary_path.exists()

    with summary_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow([
                "setting_case",
                "seed",
                "level",
                "fps",
                "duration_pi",
                "theta0_deg",
                "dz",
                "steps",
                "gamma0",
                "gamma1",
            ])

        for level in level_list:
            path = video_path(setting, level, duration_pi, fps, theta0, dz)
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing {path}. Run `python run_robustness.py generate --setting {setting}` first."
                )

            set_seed(seed)
            test_name = (
                f"single_{setting}_{duration_pi:.1f}pi_{fps}fps_"
                f"degree{theta0}_degs{dz}_level{level}_{seed}"
            )
            args = SimpleNamespace(
                data_base="pendulum",
                result_base=str(result_base),
                threshold=0.5,
                encoder="cnn",
                H=64,
                W=64,
                device=device,
                dt=1.0 / fps,
                steps=steps,
                tau=1.0,
                var_weight=1.0,
                lr_enc=1e-3,
                lr_ode=1e-2,
                gamma0_init=1.0,
                gamma1_init=1.0,
                dtheta0_deg_s=0,
                seed=seed,
                fps=fps,
                path=str(path),
                result_path=str(save_dir / f"{test_name}.png"),
                setting_case=setting,
                duration_pi=duration_pi,
                duration_seconds=duration_pi * np.pi,
                theta0_deg_case=theta0,
                dz_case=dz,
                noise_level=level,
            )
            gamma0, gamma1 = main_ablation.main(args)
            writer.writerow([setting, seed, level, fps, duration_pi, theta0, dz, steps, gamma0, gamma1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run noisy-environment robustness experiments.")
    parser.add_argument("action", choices=["generate", "train", "paper"])
    parser.add_argument("--setting", choices=["all", *GENERATOR_SCRIPTS.keys()], default="all")
    parser.add_argument("--levels", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"])
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()

    for setting in selected_settings(args.setting):
        if args.action in {"generate", "paper"} and not args.skip_generate:
            generate_setting(setting)
        if args.action in {"train", "paper"}:
            for seed in args.seeds:
                train_setting(setting, seed, args.steps, args.device, args.levels)


if __name__ == "__main__":
    main()
