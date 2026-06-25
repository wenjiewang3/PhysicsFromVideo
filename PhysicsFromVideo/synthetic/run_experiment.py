#!/usr/bin/env python3
"""Convenience runner for the paper's synthetic experiments."""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import intensity_multi
import intensity_single
import pendulum_multi
import pendulum_single
import scale_multi
import scale_single
from syn_gen import intensity_data, pendulum_data, scale_data


ROOT = Path(__file__).resolve().parent


DATASETS = {
    "pendulum": {
        "generator": "pendulum",
        "setting": "undamped",
        "gamma0": 4.0,
        "gamma1": 0.0,
        "duration_pi": 0.5,
        "fps": 6,
        "theta0_deg": 90,
        "dz_values": [0, 100, -100],
        "data_dir": ROOT / "data" / "pendulum" / "undamped",
    },
    "intensity": {
        "generator": "intensity",
        "setting": "critical",
        "gamma0": 4.0,
        "gamma1": 4.0,
        "duration": 2.0,
        "fps": 10,
        "z0": 1.0,
        "dz_values": [0, 200, -200],
        "data_dir": ROOT / "data" / "intensity" / "critical",
    },
    "scale": {
        "generator": "scale",
        "setting": "undamped",
        "gamma0": 4.0,
        "gamma1": 0.0,
        "duration": 2.0,
        "fps": 60,
        "z0": 1.0,
        "dz_values": [0.0, 10.0, -10.0],
        "data_dir": ROOT / "data" / "scale" / "undamped",
    },
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def selected_datasets(name: str) -> list[str]:
    return list(DATASETS) if name == "all" else [name]


def pendulum_filename(cfg: dict, dz: float) -> str:
    return (
        f"{cfg['duration_pi']:.1f}pi_{cfg['fps']}fps_"
        f"degree{cfg['theta0_deg']}_degs{dz}.mp4"
    )


def scalar_filename(cfg: dict, dz: float) -> str:
    return f"{cfg['duration']:.1f}_{cfg['fps']}fps_z{cfg['z0']}_dz{dz}.mp4"


def video_path(dataset: str, dz: float) -> Path:
    cfg = DATASETS[dataset]
    if dataset == "pendulum":
        return cfg["data_dir"] / pendulum_filename(cfg, dz)
    return cfg["data_dir"] / scalar_filename(cfg, dz)


def generate_dataset(dataset: str) -> None:
    cfg = DATASETS[dataset]
    cfg["data_dir"].mkdir(parents=True, exist_ok=True)

    if dataset == "pendulum":
        for dz in cfg["dz_values"]:
            args = SimpleNamespace(
                gamma0=cfg["gamma0"],
                gamma1=cfg["gamma1"],
                theta0_deg=cfg["theta0_deg"],
                dtheta0_deg_s=dz,
                duration=cfg["duration_pi"] * math.pi,
                fps=cfg["fps"],
                L=1.0,
                size=64,
                rod_px=8.0,
                bob_ratio=0.1,
                trail=0,
                pivot_y_frac=0.5,
                zoom=1.0,
                out=str(video_path(dataset, dz)),
            )
            pendulum_data.main(args)
        return

    if dataset == "intensity":
        for dz in cfg["dz_values"]:
            args = SimpleNamespace(
                gamma0=cfg["gamma0"],
                gamma1=cfg["gamma1"],
                z0=cfg["z0"],
                dz0=dz,
                duration=cfg["duration"],
                fps=cfg["fps"],
                size=64,
                z_min=0.2,
                z_max=1.0,
                shape_mode="random_polygon",
                seed=0,
                invert=False,
                out=str(video_path(dataset, dz)),
            )
            intensity_data.main(args)
        return

    if dataset == "scale":
        for dz in cfg["dz_values"]:
            args = SimpleNamespace(
                gamma0=cfg["gamma0"],
                gamma1=cfg["gamma1"],
                z0=cfg["z0"],
                dz0=dz,
                duration=cfg["duration"],
                fps=cfg["fps"],
                size=64,
                args_radius_frac=0.25,
                max_fill_frac=0.9,
                out=str(video_path(dataset, dz)),
            )
            scale_data.main(args)
        return

    raise ValueError(f"Unknown dataset: {dataset}")


def base_train_args(dataset: str, seed: int, steps: int, device: str) -> SimpleNamespace:
    cfg = DATASETS[dataset]
    return SimpleNamespace(
        threshold=0.5,
        encoder="cnn",
        H=64,
        W=64,
        device=device,
        dt=1.0 / cfg["fps"],
        steps=steps,
        tau=1.0,
        var_weight=1.0,
        lr_enc=1e-3,
        lr_ode=1e-2,
        gamma0_init=1.0,
        gamma1_init=1.0,
        seed=seed,
        fps=cfg["fps"],
    )


def train_single(dataset: str, seed: int, steps: int, device: str) -> tuple[float, float]:
    cfg = DATASETS[dataset]
    set_seed(seed)
    args = base_train_args(dataset, seed, steps, device)
    args.path = str(video_path(dataset, cfg["dz_values"][0]))
    out_dir = ROOT / "results" / "single" / dataset / cfg["setting"] / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    args.result_path = str(out_dir / f"{dataset}_single_seed{seed}.png")

    if dataset == "pendulum":
        return pendulum_single.main(args)
    if dataset == "intensity":
        return intensity_single.main(args)
    if dataset == "scale":
        return scale_single.main(args)
    raise ValueError(f"Unknown dataset: {dataset}")


def train_multi(dataset: str, seed: int, steps: int, device: str) -> tuple[float, float]:
    cfg = DATASETS[dataset]
    set_seed(seed)
    args = base_train_args(dataset, seed, steps, device)
    args.paths = ", ".join(str(video_path(dataset, dz)) for dz in cfg["dz_values"])
    out_dir = ROOT / "results" / "multi" / dataset / cfg["setting"] / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    args.result_path = str(out_dir / f"{dataset}_multi_seed{seed}.png")

    if dataset == "pendulum":
        return pendulum_multi.main(args)
    if dataset == "intensity":
        return intensity_multi.main(args)
    if dataset == "scale":
        return scale_multi.main(args)
    raise ValueError(f"Unknown dataset: {dataset}")


def append_summary(mode: str, dataset: str, seed: int, steps: int, gamma0: float, gamma1: float) -> None:
    path = ROOT / "results" / f"{mode}_summary.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["mode", "dataset", "setting", "seed", "steps", "gamma0", "gamma1"])
        writer.writerow([mode, dataset, DATASETS[dataset]["setting"], seed, steps, gamma0, gamma1])


def ensure_data(dataset: str) -> None:
    missing = [dz for dz in DATASETS[dataset]["dz_values"] if not video_path(dataset, dz).exists()]
    if missing:
        print(f"[INFO] Missing {dataset} videos for dz={missing}; generating them first.")
        generate_dataset(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Physics-from-Video synthetic experiments.")
    parser.add_argument("action", choices=["generate", "single", "multi", "paper"])
    parser.add_argument("--dataset", choices=["all", *DATASETS.keys()], default="all")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "cuda:0", "cuda:1"])
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()

    for dataset in selected_datasets(args.dataset):
        if args.action in {"generate", "paper"} and not args.skip_generate:
            generate_dataset(dataset)

        if args.action in {"single", "multi"} and not args.skip_generate:
            ensure_data(dataset)

        if args.action in {"single", "paper"}:
            for seed in args.seeds:
                gamma0, gamma1 = train_single(dataset, seed, args.steps, args.device)
                append_summary("single", dataset, seed, args.steps, gamma0, gamma1)

        if args.action in {"multi", "paper"}:
            for seed in args.seeds:
                gamma0, gamma1 = train_multi(dataset, seed, args.steps, args.device)
                append_summary("multi", dataset, seed, args.steps, gamma0, gamma1)


if __name__ == "__main__":
    main()
