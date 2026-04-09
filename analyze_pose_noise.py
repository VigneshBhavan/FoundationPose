#!/usr/bin/env python3
"""Analyze the noise profile of FoundationPose from a pose log JSON file.

Loads pose_log.json (produced by run_realsense.py), filters to tracking-only
entries, and computes translation/rotation statistics, covariances,
frame-to-frame jitter, and autocorrelation. Saves a noise_profile.json and
diagnostic plots.
"""

import argparse
import json
import os

import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


def load_tracking_poses(path):
    """Load pose log and return only tracking entries."""
    with open(path, "r") as f:
        entries = json.load(f)

    tracking = [e for e in entries if e["type"] == "track"]
    if len(tracking) == 0:
        raise ValueError("No tracking entries found in pose log. "
                         "Only 'register' entries are present.")

    frames = np.array([e["frame"] for e in tracking])
    times = np.array([e["time"] for e in tracking])
    poses = np.array([e.get("pose") or e.get("pose_cam") for e in tracking])  # (N, 4, 4)

    return frames, times, poses


def extract_translation_rotation(poses):
    """Extract translation vectors and rotation vectors from 4x4 pose matrices.

    Returns:
        translations: (N, 3) in meters
        rotvecs: (N, 3) in radians
    """
    translations = poses[:, :3, 3]  # (N, 3)
    rot_matrices = poses[:, :3, :3]  # (N, 3, 3)
    rotvecs = Rotation.from_matrix(rot_matrices).as_rotvec()  # (N, 3)
    return translations, rotvecs


def compute_autocorrelation(signal, max_lag):
    """Compute normalized autocorrelation of a 1D signal at lags 1..max_lag.

    Returns an array of length max_lag with autocorrelation values.
    """
    n = len(signal)
    if n < max_lag + 1:
        return np.full(max_lag, np.nan)

    mean = np.mean(signal)
    centered = signal - mean
    var = np.sum(centered ** 2)
    if var == 0:
        return np.zeros(max_lag)

    autocorr = np.empty(max_lag)
    for lag in range(1, max_lag + 1):
        autocorr[lag - 1] = np.sum(centered[:n - lag] * centered[lag:]) / var
    return autocorr


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FoundationPose noise profile from a pose log.")
    parser.add_argument("--pose_log", type=str, required=True,
                        help="Path to pose_log.json produced by run_realsense.py")
    parser.add_argument("--output", type=str, default="noise_profile.json",
                        help="Path for the output noise profile JSON "
                             "(default: noise_profile.json)")
    parser.add_argument("--skip_seconds", type=float, default=0.0,
                        help="Skip the first N seconds of tracking data "
                             "(use to trim initial convergence period)")
    args = parser.parse_args()

    # ---- Load data ----
    frames, times, poses = load_tracking_poses(args.pose_log)

    # Trim initial convergence period
    if args.skip_seconds > 0:
        t0 = times[0]
        keep = times >= (t0 + args.skip_seconds)
        frames = frames[keep]
        times = times[keep]
        poses = poses[keep]
        print(f"Skipped first {args.skip_seconds}s — "
              f"{keep.sum()} of {len(keep)} frames remaining")

    n = len(frames)
    if n == 0:
        print("No frames remaining after skip. Reduce --skip_seconds.")
        return
    duration = times[-1] - times[0] if n > 1 else 0.0

    translations, rotvecs = extract_translation_rotation(poses)

    # ---- Per-axis statistics ----
    t_mean = np.mean(translations, axis=0)  # (3,)
    t_std = np.std(translations, axis=0)    # (3,)
    r_mean = np.mean(rotvecs, axis=0)       # (3,)
    r_std = np.std(rotvecs, axis=0)         # (3,)

    # ---- Covariance matrices ----
    t_cov = np.cov(translations, rowvar=False)  # (3, 3)
    r_cov = np.cov(rotvecs, rowvar=False)       # (3, 3)

    # Handle single-frame edge case where cov returns scalar 0
    if t_cov.ndim == 0:
        t_cov = np.zeros((3, 3))
    if r_cov.ndim == 0:
        r_cov = np.zeros((3, 3))

    # ---- Frame-to-frame deltas ----
    t_deltas = np.diff(translations, axis=0)  # (N-1, 3)
    r_deltas = np.diff(rotvecs, axis=0)       # (N-1, 3)

    t_delta_norms = np.linalg.norm(t_deltas, axis=1)  # (N-1,)
    r_delta_norms = np.linalg.norm(r_deltas, axis=1)  # (N-1,)

    if len(t_delta_norms) > 0:
        t_delta_mean = float(np.mean(t_delta_norms))
        t_delta_std = float(np.std(t_delta_norms))
        r_delta_mean = float(np.mean(r_delta_norms))
        r_delta_std = float(np.std(r_delta_norms))
    else:
        t_delta_mean = t_delta_std = 0.0
        r_delta_mean = r_delta_std = 0.0

    # ---- Autocorrelation of translation deltas (lags 1-5) ----
    max_lag = 5
    autocorr_x = compute_autocorrelation(t_deltas[:, 0] if len(t_deltas) > 0
                                         else np.array([]), max_lag)
    autocorr_y = compute_autocorrelation(t_deltas[:, 1] if len(t_deltas) > 0
                                         else np.array([]), max_lag)
    autocorr_z = compute_autocorrelation(t_deltas[:, 2] if len(t_deltas) > 0
                                         else np.array([]), max_lag)

    # ---- Print summary ----
    print("=" * 65)
    print("  FoundationPose Noise Profile Analysis")
    print("=" * 65)
    print(f"  Tracking frames:  {n}")
    print(f"  Duration:         {duration:.2f} s")
    print(f"  Avg frame rate:   {(n - 1) / duration:.1f} Hz" if duration > 0
          else "  Avg frame rate:   N/A (single frame)")
    print()

    axis_labels = ["x", "y", "z"]

    print("  Translation (meters):")
    print(f"    {'axis':<5} {'mean':>12} {'std':>12}")
    for i, ax in enumerate(axis_labels):
        print(f"    {ax:<5} {t_mean[i]:>12.6f} {t_std[i]:>12.6f}")
    print()

    print("  Rotation (radians, rotvec):")
    print(f"    {'axis':<5} {'mean':>12} {'std':>12}")
    for i, ax in enumerate(axis_labels):
        print(f"    {ax:<5} {r_mean[i]:>12.6f} {r_std[i]:>12.6f}")
    print()

    print("  Translation covariance (3x3):")
    for row in t_cov:
        print("    " + "  ".join(f"{v:>12.2e}" for v in row))
    print()

    print("  Rotation covariance (3x3):")
    for row in r_cov:
        print("    " + "  ".join(f"{v:>12.2e}" for v in row))
    print()

    print("  Frame-to-frame deltas:")
    print(f"    Translation norm:  mean={t_delta_mean:.6f} m, "
          f"std={t_delta_std:.6f} m")
    print(f"    Rotation norm:     mean={r_delta_mean:.6f} rad "
          f"({np.degrees(r_delta_mean):.4f} deg), "
          f"std={r_delta_std:.6f} rad ({np.degrees(r_delta_std):.4f} deg)")
    print()

    print("  Autocorrelation of translation deltas (lags 1-5):")
    print(f"    {'lag':<5} {'x':>10} {'y':>10} {'z':>10}")
    for lag in range(max_lag):
        print(f"    {lag + 1:<5} {autocorr_x[lag]:>10.4f} "
              f"{autocorr_y[lag]:>10.4f} {autocorr_z[lag]:>10.4f}")
    print("=" * 65)

    # ---- Save noise profile JSON ----
    profile = {
        "t_mean": t_mean.tolist(),
        "t_std": t_std.tolist(),
        "r_mean": r_mean.tolist(),
        "r_std": r_std.tolist(),
        "t_cov": t_cov.tolist(),
        "r_cov": r_cov.tolist(),
        "n_frames": n,
        "duration_seconds": duration,
        "frame_to_frame": {
            "translation_delta_mean": t_delta_mean,
            "translation_delta_std": t_delta_std,
            "rotation_delta_mean": r_delta_mean,
            "rotation_delta_std": r_delta_std,
        },
        "autocorrelation_translation_deltas": {
            "lags": list(range(1, max_lag + 1)),
            "x": autocorr_x.tolist(),
            "y": autocorr_y.tolist(),
            "z": autocorr_z.tolist(),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"\nNoise profile saved to {args.output}")

    # ---- Plotting ----
    output_dir = os.path.dirname(os.path.abspath(args.output))
    relative_times = times - times[0]

    # Figure 1: Translation over time
    fig1, axes1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig1.suptitle("Translation Over Time")
    for i, ax_label in enumerate(axis_labels):
        axes1[i].plot(relative_times, translations[:, i], linewidth=0.8)
        axes1[i].axhline(t_mean[i], color="r", linestyle="--", linewidth=0.7,
                         label=f"mean={t_mean[i]:.4f}")
        axes1[i].set_ylabel(f"{ax_label} (m)")
        axes1[i].legend(loc="upper right", fontsize=8)
        axes1[i].grid(True, alpha=0.3)
    axes1[-1].set_xlabel("Time (s)")
    fig1.tight_layout()
    fig1_path = os.path.join(output_dir, "translation_over_time.png")
    fig1.savefig(fig1_path, dpi=150)
    print(f"Saved {fig1_path}")

    # Figure 2: Rotation over time (degrees)
    rotvecs_deg = np.degrees(rotvecs)
    r_mean_deg = np.degrees(r_mean)

    fig2, axes2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig2.suptitle("Rotation Over Time (rotvec, degrees)")
    rot_labels = ["rx", "ry", "rz"]
    for i, ax_label in enumerate(rot_labels):
        axes2[i].plot(relative_times, rotvecs_deg[:, i], linewidth=0.8)
        axes2[i].axhline(r_mean_deg[i], color="r", linestyle="--", linewidth=0.7,
                         label=f"mean={r_mean_deg[i]:.2f} deg")
        axes2[i].set_ylabel(f"{ax_label} (deg)")
        axes2[i].legend(loc="upper right", fontsize=8)
        axes2[i].grid(True, alpha=0.3)
    axes2[-1].set_xlabel("Time (s)")
    fig2.tight_layout()
    fig2_path = os.path.join(output_dir, "rotation_over_time.png")
    fig2.savefig(fig2_path, dpi=150)
    print(f"Saved {fig2_path}")

    # Figure 3: Translation residual histograms
    t_residuals = translations - t_mean  # (N, 3)

    fig3, axes3 = plt.subplots(1, 3, figsize=(12, 4))
    fig3.suptitle("Translation Residuals from Mean")
    for i, ax_label in enumerate(axis_labels):
        axes3[i].hist(t_residuals[:, i], bins=50, edgecolor="black",
                      linewidth=0.5, alpha=0.7)
        axes3[i].set_xlabel(f"{ax_label} residual (m)")
        axes3[i].set_ylabel("Count")
        axes3[i].set_title(f"{ax_label}: std={t_std[i]:.5f} m")
        axes3[i].grid(True, alpha=0.3)
    fig3.tight_layout()
    fig3_path = os.path.join(output_dir, "translation_residual_hist.png")
    fig3.savefig(fig3_path, dpi=150)
    print(f"Saved {fig3_path}")

    # Figure 4: Rotation residual histograms (degrees)
    r_residuals_deg = np.degrees(rotvecs - r_mean)  # (N, 3)
    r_std_deg = np.degrees(r_std)

    fig4, axes4 = plt.subplots(1, 3, figsize=(12, 4))
    fig4.suptitle("Rotation Residuals from Mean (degrees)")
    for i, ax_label in enumerate(rot_labels):
        axes4[i].hist(r_residuals_deg[:, i], bins=50, edgecolor="black",
                      linewidth=0.5, alpha=0.7)
        axes4[i].set_xlabel(f"{ax_label} residual (deg)")
        axes4[i].set_ylabel("Count")
        axes4[i].set_title(f"{ax_label}: std={r_std_deg[i]:.3f} deg")
        axes4[i].grid(True, alpha=0.3)
    fig4.tight_layout()
    fig4_path = os.path.join(output_dir, "rotation_residual_hist.png")
    fig4.savefig(fig4_path, dpi=150)
    print(f"Saved {fig4_path}")

    plt.show()


if __name__ == "__main__":
    main()
