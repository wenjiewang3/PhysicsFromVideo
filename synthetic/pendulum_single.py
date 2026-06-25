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
        x = load_bw_videos_same_shape(paths, binarize=True,
                                      threshold=args.threshold, device=device)  # (B,T,1,H,W)
    else:
        vid = load_bw_video_to_tensor(args.path, binarize=True,
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
            stats = estimate_z0_and_derivs(z, dt)   # z is (1,T,1)
            print(
                f"[{step:04d}] loss={loss.item():.4e} "
                f"| gamma0={gamma0.item():.4f} gamma1={gamma1.item():.4f} "
                f"| z(0)={stats['z0']:.4f}  z'(0)≈{stats['z0p']:.4e}  z''(0)≈{stats['z0pp']:.4e} z(mid)={z[0][int(len(z[0])/2)].item():.4e}"
            )


    # --- Evaluate and plot per video ---
    enc.eval()
    with torch.no_grad():
        z_eval = enc(x).squeeze(-1)  # (B,T)

    # 批量估计 z0, z0p
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



if __name__ == "__main__":
    print("------------------------")
    ap = argparse.ArgumentParser(description="Learning ODE parameters from single video")

    # Dataset and Result paths
    ap.add_argument("--data_base", type=str, default="pendulum",
                    help="Base folder for dataset (e.g. 'pendulum')")
    ap.add_argument("--result_base", type=str, default="Result/pendulum_single",
                    help="Base folder for results")

    # Model & Preprocessing
    ap.add_argument("--threshold", type=float, default=0.5, help="binarize threshold in [0,1]")
    ap.add_argument("--encoder", type=str, default="cnn", choices=["cnn","mlp","intensity"], help="which encoder to use")
    ap.add_argument("--H", type=int, default=64, help="height of video (for MLP)")
    ap.add_argument("--W", type=int, default=64, help="width  of video (for MLP)")
    ap.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda","cuda:0","cuda:1"],
                    help="compute device (default: auto)")
    ap.add_argument("--dt", type=float, default=1/20, help="dt (overwritten by loop)")

    # Training Hyperparameters
    ap.add_argument("--steps", type=int, default=30000, help="training steps")
    ap.add_argument("--tau", type=float, default=1.0, help="lower bound on std of z (0.5~1)")
    ap.add_argument("--var_weight", type=float, default=1.0, help="weight for variance floor term")

    ap.add_argument("--lr_enc", type=float, default=1e-3, help="learning rate for encoder")
    ap.add_argument("--lr_ode", type=float, default=1e-2, help="learning rate for (gamma0, gamma1)")

    # Initialization
    ap.add_argument("--gamma0_init", type=float, default=1.0, help="initial value of gamma0")
    ap.add_argument("--gamma1_init", type=float, default=1.0, help="initial value of gamma1")
    ap.add_argument("--dtheta0_deg_s", type=float, default=0, help="initial angular velocity")

    ap.add_argument("--seed", type=int, default=42, help="random seed")

    args = ap.parse_args()

    # Define the experiment loops
    setting_cases = ["undamped"]
    fps_list = [6]              # e.g. [10, 20, 60]
    seconds_list = [0.5]
    degree_list = [90]
    dz_list = [0, 100, -100]
    args.data_path = "syn_gen/pendulum_dataset"
    result_folder = "result/syn/pendulum_intensity" # Corresponds to multi_intensity

    for setting_case in setting_cases:
        # Create result directory for this seed/setting
        save_dir = os.path.join(args.result_base, f"{setting_case}_new", f"seed{args.seed}")
        os.makedirs(save_dir, exist_ok=True)

        # path for the aggregate summary CSV
        summary_path = os.path.join(
            args.result_base,
            f"{setting_case}_summary_seed{args.seed}.txt"
        )

        # Write header if new
        if not os.path.exists(summary_path):
            with open(summary_path, "w") as f:
                f.write("setting_case,seed,fps,second,theta0_deg,dz,encoder,tau,lr_enc,lr_ode,steps,gamma0,gamma1\n")

        for args.fps in fps_list:
            args.dt = 1.0 / args.fps

            for second in seconds_list:
                for theta0_deg in degree_list:
                    for dz in dz_list:
                        # Construct file name and paths
                        video_filename = f"{second:.1f}pi_{int(args.fps)}fps_degree{theta0_deg}_degs{dz}.mp4"

                        # Data path: e.g. pendulum/undamped/0.7pi_20fps_degree10_degs0.mp4
                        args.path = os.path.join(args.data_path, setting_case, video_filename)

                        # Experiment ID
                        test_name = f"single_{setting_case}_{second:.1f}pi_{int(args.fps)}fps_degree{theta0_deg}_degs{dz}_{args.seed}"

                        # Path to save the plot/result image
                        args.result_path = os.path.join(save_dir, f"{test_name}.png")

                        # Log file
                        log_path = os.path.join(save_dir, f"log_{test_name}.txt")

                        # Redirect stdout/stderr to file and console
                        tee = TeeLogger(log_path)
                        sys.stdout = tee
                        sys.stderr = tee

                        print(
                            f"Setting: {setting_case} | Seed: {args.seed} | Second: {second:.2f}pi | "
                            f"Theta: {theta0_deg} | dz {dz} | FPS: {int(args.fps)}\n"
                            f"Test Name: {test_name}\n"
                            f"Params: tau={args.tau} | steps={args.steps} | "
                            f"gamma0_init={args.gamma0_init} | gamma1_init={args.gamma1_init} | "
                            f"lr_enc={args.lr_enc} | lr_ode={args.lr_ode}"
                        )

                        # --- Run Main Execution ---
                        # Ensure main() accepts args and returns the estimated parameters
                        gamma0, gamma1 = main(args)

                        print("---------")

                        # Close logger and restore standard output
                        tee.close()
                        sys.stdout = sys.__stdout__
                        sys.stderr = sys.__stderr__

                        # Append results to summary file
                        with open(summary_path, "a") as f:
                            f.write(
                                f"{setting_case},{args.seed},{int(args.fps)},{second:.2f}pi,"
                                f"{theta0_deg}, dz {dz}, {args.encoder},{args.tau},{args.lr_enc},{args.lr_ode},"
                                f"{args.steps},{gamma0},{gamma1}\n"
                            )
