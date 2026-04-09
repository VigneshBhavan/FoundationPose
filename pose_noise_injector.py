#!/usr/bin/env python3
"""Inject FoundationPose-like noise into ground-truth poses.

Loads a noise profile JSON (produced by analyze_pose_noise.py) and provides
several noise injection modes suitable for sim-to-real transfer in Isaac Sim
or any other simulator.

Usage as a library:
    from pose_noise_injector import PoseNoiseInjector

    injector = PoseNoiseInjector("noise_profile.json", scale=1.5)
    injector.reset()

    for gt_pose in episode_poses:
        noisy = injector.add_noise_realistic(gt_pose)
"""

import argparse
import json

import numpy as np
from scipy.spatial.transform import Rotation


class OrnsteinUhlenbeckProcess:
    """Ornstein-Uhlenbeck process for generating temporally correlated noise.

    The process mean-reverts toward `mean` with rate `theta`, driven by
    Gaussian increments scaled by `std` and timestep `dt`.

    Parameters
    ----------
    mean : np.ndarray
        Long-term mean the process reverts to (shape determines dimensionality).
    std : np.ndarray
        Standard deviation of the driving noise (same shape as mean).
    theta : float
        Mean-reversion rate.  Higher values produce faster decorrelation.
    dt : float
        Simulation timestep in seconds.
    """

    def __init__(self, mean, std, theta=0.15, dt=1 / 30):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.theta = theta
        self.dt = dt
        self.state = np.zeros_like(self.mean)

    def sample(self):
        """Advance one timestep and return the current noise sample."""
        noise = np.random.normal(size=self.state.shape)
        self.state += (
            self.theta * (self.mean - self.state) * self.dt
            + self.std * np.sqrt(self.dt) * noise
        )
        return self.state.copy()

    def reset(self):
        """Reset the process state to zero."""
        self.state = np.zeros_like(self.mean)


class PoseNoiseInjector:
    """Add FoundationPose-like noise to ground-truth 4x4 poses.

    Three noise modes are provided in order of increasing realism:

    * ``add_noise_iid`` -- independent Gaussian noise every frame.
    * ``add_noise_correlated`` -- temporally smooth noise via an
      Ornstein-Uhlenbeck process.  Call ``reset()`` at the start of each
      episode.
    * ``add_noise_realistic`` -- OU noise with occasional large outlier
      jumps (5 % probability), mimicking the sporadic tracking failures
      seen in real FoundationPose output.

    Parameters
    ----------
    noise_profile_path : str
        Path to a ``noise_profile.json`` produced by ``analyze_pose_noise.py``.
    scale : float
        Multiplier on all noise magnitudes.  Use 1.5--2.0 to over-estimate
        noise for robustness during training.
    """

    def __init__(self, noise_profile_path, scale=1.0):
        with open(noise_profile_path, "r") as f:
            profile = json.load(f)

        self.scale = float(scale)

        # Per-axis standard deviations (3,)
        self.t_std = np.array(profile["t_std"], dtype=np.float64) * self.scale
        self.r_std = np.array(profile["r_std"], dtype=np.float64) * self.scale

        # Covariance matrices (3, 3) -- stored for potential future use
        self.t_cov = np.array(profile["t_cov"], dtype=np.float64) * (self.scale ** 2)
        self.r_cov = np.array(profile["r_cov"], dtype=np.float64) * (self.scale ** 2)

        # Frame-to-frame delta stats (used by OU process)
        ftf = profile.get("frame_to_frame", {})
        self.t_delta_std = float(ftf.get("translation_delta_std", 0.0)) * self.scale
        self.r_delta_std = float(ftf.get("rotation_delta_std", 0.0)) * self.scale

        # OU processes for translation (3-D) and rotation (3-D)
        self._ou_translation = OrnsteinUhlenbeckProcess(
            mean=np.zeros(3), std=self.t_std, theta=0.15, dt=1 / 30
        )
        self._ou_rotation = OrnsteinUhlenbeckProcess(
            mean=np.zeros(3), std=self.r_std, theta=0.15, dt=1 / 30
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Reset internal OU state.  Call at the start of each episode."""
        self._ou_translation.reset()
        self._ou_rotation.reset()

    def add_noise_iid(self, gt_pose):
        """Add independent Gaussian noise to a 4x4 pose.

        Translation noise is additive.  Rotation noise is a small random
        rotation (axis-angle / rotvec) left-multiplied onto the rotation
        part of the pose.

        Parameters
        ----------
        gt_pose : np.ndarray
            (4, 4) homogeneous transformation matrix.

        Returns
        -------
        np.ndarray
            (4, 4) noisy pose.
        """
        gt_pose = np.array(gt_pose, dtype=np.float64)
        t_noise = np.random.normal(0.0, self.t_std)
        r_noise = np.random.normal(0.0, self.r_std)
        return self._apply_noise(gt_pose, t_noise, r_noise)

    def add_noise_correlated(self, gt_pose):
        """Add temporally correlated noise via the Ornstein-Uhlenbeck process.

        Must call ``reset()`` at the start of each episode so that the OU
        state does not carry over.

        Parameters
        ----------
        gt_pose : np.ndarray
            (4, 4) homogeneous transformation matrix.

        Returns
        -------
        np.ndarray
            (4, 4) noisy pose.
        """
        gt_pose = np.array(gt_pose, dtype=np.float64)
        t_noise = self._ou_translation.sample()
        r_noise = self._ou_rotation.sample()
        return self._apply_noise(gt_pose, t_noise, r_noise)

    def add_noise_realistic(self, gt_pose):
        """OU noise with occasional large outlier jumps.

        With 5 % probability the noise is amplified by 3x (simulating a
        tracking glitch).  Otherwise behaves identically to
        ``add_noise_correlated``.

        Parameters
        ----------
        gt_pose : np.ndarray
            (4, 4) homogeneous transformation matrix.

        Returns
        -------
        np.ndarray
            (4, 4) noisy pose.
        """
        gt_pose = np.array(gt_pose, dtype=np.float64)

        if np.random.rand() < 0.05:
            # Outlier jump: 3x the per-axis std, drawn independently
            t_noise = np.random.normal(0.0, self.t_std * 3.0)
            r_noise = np.random.normal(0.0, self.r_std * 3.0)
        else:
            t_noise = self._ou_translation.sample()
            r_noise = self._ou_rotation.sample()

        return self._apply_noise(gt_pose, t_noise, r_noise)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_noise(pose, t_noise, r_noise):
        """Apply translation and rotation noise to a 4x4 pose.

        Translation noise is added directly.  Rotation noise is interpreted
        as a rotation vector (axis * angle) and left-multiplied onto the
        existing rotation:  R_noisy = R_noise @ R_gt.
        """
        noisy = pose.copy()

        # Translation: additive
        noisy[:3, 3] += t_noise

        # Rotation: left-multiply small rotation
        delta_rot = Rotation.from_rotvec(r_noise).as_matrix()  # (3, 3)
        noisy[:3, :3] = delta_rot @ pose[:3, :3]

        return noisy


# ======================================================================
# Demo / self-test
# ======================================================================

def _demo(noise_profile_path):
    """Run a quick demo of all three noise modes and print summary stats."""

    injector = PoseNoiseInjector(noise_profile_path, scale=1.5)
    gt_pose = np.eye(4)
    n_steps = 100

    modes = {
        "iid": injector.add_noise_iid,
        "correlated": injector.add_noise_correlated,
        "realistic": injector.add_noise_realistic,
    }

    for mode_name, noise_fn in modes.items():
        injector.reset()
        translations = np.empty((n_steps, 3))

        for i in range(n_steps):
            noisy = noise_fn(gt_pose)
            translations[i] = noisy[:3, 3]

        mean = np.mean(translations, axis=0)
        std = np.std(translations, axis=0)
        abs_max = np.max(np.abs(translations), axis=0)

        print(f"--- {mode_name} ({n_steps} steps, scale=1.5) ---")
        print(f"  translation mean:    [{mean[0]:+.6f}, {mean[1]:+.6f}, {mean[2]:+.6f}]")
        print(f"  translation std:     [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
        print(f"  translation abs max: [{abs_max[0]:.6f}, {abs_max[1]:.6f}, {abs_max[2]:.6f}]")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo: inject FoundationPose-like noise into a dummy pose."
    )
    parser.add_argument(
        "--noise_profile",
        type=str,
        required=True,
        help="Path to noise_profile.json produced by analyze_pose_noise.py",
    )
    args = parser.parse_args()
    _demo(args.noise_profile)
