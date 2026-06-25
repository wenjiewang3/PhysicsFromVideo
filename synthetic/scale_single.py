import argparse
import torch
from data_loader import load_bw_video_to_tensor, load_bw_videos_same_shape
from encoder import FrameEncoderCNN, FrameEncoderMLP
import torch.nn as nn
import torch.optim as optim
from loss import ode_residual_loss, ode_residual_loss_centered, ode_residual_loss_backward
from plot_tools import plot_est_vs_true
from utils import estimate_z0_and_derivs, estimate_z0_and_derivs_batch
import os
import sys
from utils import TeeLogger

def main(args):
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device.startswith("cuda"):
            assert torch.cuda.is_available(), "CUDA not available"
        device = torch.device(args.device)

    # --- Load one or many videos ---
    multi_video = False
    if multi_video:
        paths = [p.strip() for p in args.paths.split(",") if p.strip()]
        x = load_bw_videos_same_shape(paths, binarize=False,
                                      threshold=args.threshold, device=device)  # (B,T,1,H,W)
    else:
        vid = load_bw_video_to_tensor(args.path, binarize=False,
                                      threshold=args.threshold, device=device) # (T,1,H,W)
        x = vid.unsqueeze(0)  # (1,T,1,H,W)

    B, T, _, H, W = x.shape
    print("Loaded batch:", x.shape)


    if args.encoder == "cnn":
        enc = FrameEncoderCNN().to(device)
    else:
        enc = FrameEncoderMLP(args.H, args.W).to(device)

    # ODE parameters
    gamma0 = nn.Parameter(torch.tensor(args.gamma0_init, device=device), requires_grad=True)
    gamma1 = nn.Parameter(torch.tensor(args.gamma1_init, device=device), requires_grad=True)

    optimizer = optim.Adam([
        {"params": enc.parameters(), "lr": args.lr_enc},
        {"params": [gamma0, gamma1], "lr": args.lr_ode},
    ])

    dt = float(args.dt)
    enc.train()
    for step in range(1, args.steps + 1):
        z = enc(x)
        # loss = ode_residual_loss(z, dt, gamma0, gamma1)
        loss = ode_residual_loss_centered(z, dt, gamma0, gamma1,
        tau=args.tau, var_weight=args.var_weight)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % 2000 == 0 or step == 1 or step == args.steps:
            stats = estimate_z0_and_derivs(z, dt)   # z 是 (1,T,1)
            print(
                f"[{step:04d}] loss={loss.item():.4e} "
                f"| gamma0={gamma0.item():.4f} gamma1={gamma1.item():.4f} "
                f"| z(0)={stats['z0']:.4f}  z'(0)≈{stats['z0p']:.4e}  z''(0)≈{stats['z0pp']:.4e} z(mid)={z[0][int(len(z[0])/2)].item():.4e}"
            )

    # enc.train()
    # for step in range(1, args.steps + 1):
    #     z = enc(x)                        # (B,T,1)  keep batch dimension!
    #     loss, g0_hat_b, g1_hat_b = profiled_ode_loss_v2(
    #         z, dt=args.dt, ridge=1e-4, alpha=0.1, huber_delta=0.0
    #     )
    #     optimizer.zero_grad(set_to_none=True)
    #     loss.backward()
    #     optimizer.step()

    #     if step % 50 == 0 or step == 1 or step == args.steps:
    #         # diagnostics to verify no-collapse
    #         with torch.no_grad():
    #             z_std = z.std(dim=(1,2)).mean().item()
    #             z_mean = z.mean(dim=(1,2)).mean().item()
    #             print(f"[{step:04d}] loss={loss.item():.4e} | z.std={z_std:.3e} z.mean={z_mean:.3e} "
    #                 f"| g0_hat≈{g0_hat_b.mean().item():.4f} g1_hat≈{g1_hat_b.mean().item():.4f}")

    # --- Evaluate and plot per video ---
    enc.eval()
    with torch.no_grad():
        z_eval = enc(x).squeeze(-1)  # (B,T)

    est_stats = estimate_z0_and_derivs_batch(z_eval, dt, use_smoothing=True)
    z0_hat_all  = est_stats["z0"].cpu().numpy()
    z1_hat_all  = est_stats["z0p"].cpu().numpy()

    gamma0_hat = float(gamma0.detach().cpu())
    gamma1_hat = float(gamma1.detach().cpu())

    # true parameters
    gamma0_true, gamma1_true = 4.0016, 0.08
    z0_true, z0_grad_true = 1.0, 0.0

    for b in range(B):
        z_b = z_eval[b].detach().cpu().numpy()
        plot_est_vs_true(
            z_est=z_b, dt=dt,
            gamma0=gamma0_true, gamma1=gamma1_true, z0=z0_true, z1=z0_grad_true,
            save_path=(args.result_path if B==1 else args.result_path.replace(".png", f"_vid{b}.png")),
            gamma0_hat=gamma0_hat, gamma1_hat=gamma1_hat,
            z0_hat=float(z0_hat_all[b]),
            z1_hat=float(z1_hat_all[b]),
        )

    return round(gamma0_hat, 4), round(gamma1_hat, 4)

    # with torch.no_grad():
    #     z_eval = enc(x).squeeze(0)   # (T,1)
    # plot_z_over_time(z_eval, dt=dt, save_path="result/z_curve.png", title="Estimated z(t)")

    # estimated parameters and initial value
    # est_stats = estimate_z0_and_derivs(z_eval, dt)
    # z0_hat = round(est_stats["z0"], 5)
    # z1_hat = round(est_stats["z0p"], 5)
    # gamma0_hat = round(float(gamma0.item()), 5)
    # gamma1_hat = round(float(gamma1.item()), 5)

    # # true parameters
    # gamma0_true, gamma1_true = 4.0016, 0.08
    # z0_true, z0_grad_true = 1.0, 0.0
    # plot_est_vs_true(
    #     z_est=z_eval, dt=dt,
    #     gamma0=gamma0_true, gamma1=gamma1_true, z0=z0_true, z1=z0_grad_true,
    #     save_path=args.result_path, match_ylims=False,
    #     gamma0_hat=gamma0_hat, gamma1_hat=gamma1_hat,
    #     z0_hat=z0_hat, z1_hat=z1_hat,
    # )


if __name__ == "__main__":
    print("------------------------")
    ap = argparse.ArgumentParser(description="Learning ODE parameters from single video")


    ap.add_argument("--path", type=str, default=f"generated_video/test.mp4",
                    help="path to mp4")

    ap.add_argument("--result_path", type=str, default=f"result/setting/test.png",
                    help="path to save result image")

    ap.add_argument("--threshold", type=float, default=0.5, help="binarize threshold in [0,1]")
    ap.add_argument("--encoder", type=str, default="cnn", choices=["cnn","mlp"], help="which encoder to use")
    ap.add_argument("--H", type=int, default=64, help="height of video (for MLP)")
    ap.add_argument("--W", type=int, default=64, help="width  of video (for MLP)")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","cuda:0","cuda:1"],
                    help="compute device (default: auto)")
    ap.add_argument("--dt", type=float, default=1/20, help="dt")

    ap.add_argument("--steps", type=int, default=30000, help="training steps")
    ap.add_argument("--tau", type=float, default=1.0, help="lower bound on std of z (0.5~1)")
    ap.add_argument("--var_weight", type=float, default=1.0, help="weight for variance floor term")

    ap.add_argument("--lr_enc", type=float, default=1e-3, help="learning rate for encoder")
    ap.add_argument("--lr_ode", type=float, default=1e-2, help="learning rate for (gamma0, gamma1)")

    ap.add_argument("--gamma0_init", type=float, default=1.0, help="intial value of gamma0")
    ap.add_argument("--gamma1_init", type=float, default=1.0, help="intial value of gamma1")

    ap.add_argument("--seed", type=int, default=42, help="seed")
    args = ap.parse_args()

    args.data_path = "syn_gen/scale_dataset"
    result_folder = "result/syn/single_scale"


    for setting_case in ["undamped"]:


        summary_path = os.path.join(
            result_folder,
            f"{setting_case}_summary.txt"
        )

        os.makedirs(os.path.join(result_folder, setting_case), exist_ok=True)

        if not os.path.exists(summary_path):
            with open(summary_path, "w") as f:
                f.write("setting_case,seed,fps,second,z0,dz,encoder,tau,lr-enc,lr-ode,steps,gamma0,gamma1\n")

        second = 2.0
        z0 = 1.0
        dz_list = [0.0]
        for args.fps in [60]:
            args.dt = 1 / args.fps
            for args.seed in range(0, 5, 1):

                current_seed_folder = f"{result_folder}/{setting_case}/seed{args.seed}"
                os.makedirs(current_seed_folder, exist_ok=True)

                for dz in dz_list:
                    test_name = f"single_{second:.1f}_{args.fps}fps_z{z0}_dz{dz}"

                    args.path = f"{args.data_path}/{setting_case}/{second:.1f}_{args.fps}fps_z{z0}_dz{dz}.mp4"
                    args.result_path = f"{current_seed_folder}/{test_name}.png"

                    log_path = os.path.join(current_seed_folder, f"log_{test_name}.txt")

                    tee = TeeLogger(log_path)
                    sys.stdout = tee
                    sys.stderr = tee

                    print(
                        f"setting {setting_case}, seed {args.seed}, second: {second} "
                        f"test_name: {test_name}, tau: {args.tau}, total epoch: {args.steps} | "
                        f"gamma0_init={args.gamma0_init} | gamma1_init={args.gamma1_init} | "
                        f"lr_enc={args.lr_enc} | lr_ode={args.lr_ode} | steps={args.steps} | dt={args.dt}"
                    )

                    gamma0, gamma1 = main(args)

                    print("---------")
                    tee.close()
                    sys.stdout = sys.__stdout__
                    sys.stderr = sys.__stderr__

                    with open(summary_path, "a") as f:
                        f.write(
                            f"{setting_case},{args.seed},{int(args.fps)},{second},"
                            f"{z0}, {dz}, {args.encoder}, {args.tau}, {args.lr_enc}, {args.lr_ode}, {args.steps}, {gamma0},{gamma1}\n"
                        )