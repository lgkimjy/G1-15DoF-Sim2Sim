#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def get_log_file_path(directory):
    path = Path(directory).expanduser()
    if path.is_file():
        return path, path.parent
    return path / "stateData.h5", path


def read_h5_dataset(h5_file, key):
    if key not in h5_file:
        available = []
        h5_file.visititems(
            lambda name, item: available.append(name)
            if hasattr(item, "shape")
            else None
        )
        raise KeyError(
            f"Dataset '{key}' not found. Available datasets: "
            f"{', '.join(available) if available else '(none)'}"
        )

    data = h5_file[key][:]
    if data.ndim == 2 and data.shape[1] == 1:
        data = data[:, 0]
    return data


def reshape_vec3(data):
    if data.ndim == 1:
        data = data.reshape(-1, 3)
    return data


def reshape_rot_mat(data):
    if data.ndim == 1:
        data = data.reshape(-1, 9)
    return data.reshape((-1, 3, 3)).transpose(0, 2, 1)


def main():
    parser = argparse.ArgumentParser(description="Plot robot velocity tracking result")
    parser.add_argument("directory", type=str, help="Directory containing stateData.h5 file")
    parser.add_argument("-i", "--initial_timestamp", type=int, default=0, help="Initial timestamp")
    parser.add_argument("-e", "--end_timestamp", type=int, default=None, help="End timestamp")
    parser.add_argument(
        "--frame",
        choices=["base", "world"],
        default="base",
        help="Frame used for velocity comparison",
    )
    args = parser.parse_args()

    try:
        import h5py
    except ModuleNotFoundError:
        raise SystemExit("h5py is required. Install it with `pip install h5py`.")

    log_file_path, log_dir = get_log_file_path(args.directory)

    assets_dir = log_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(log_file_path, "r") as f:
            pdot_B = reshape_vec3(read_h5_dataset(f, "fbk/pdot_B"))
            varphi_B = reshape_vec3(read_h5_dataset(f, "fbk/varphi_B"))
            omega_B = reshape_vec3(read_h5_dataset(f, "fbk/omega_B"))
            R_B = reshape_rot_mat(read_h5_dataset(f, "fbk/R_B"))
            lin_vel_d = reshape_vec3(read_h5_dataset(f, "ctrl/lin_vel_d"))
            ang_vel_d = reshape_vec3(read_h5_dataset(f, "ctrl/ang_vel_d"))
    except OSError as e:
        raise SystemExit(
            f"Failed to open {log_file_path}: {e}\n"
            "This usually means the HDF5 log was not closed/flushed cleanly. "
            "Close the simulator window normally and use the newly created log."
        )

    T = min(
        pdot_B.shape[0],
        varphi_B.shape[0],
        omega_B.shape[0],
        R_B.shape[0],
        lin_vel_d.shape[0],
        ang_vel_d.shape[0],
    )
    pdot_B = pdot_B[:T]
    varphi_B = varphi_B[:T]
    omega_B = omega_B[:T]
    R_B = R_B[:T]
    lin_vel_d = lin_vel_d[:T]
    ang_vel_d = ang_vel_d[:T]
    t = np.arange(T)

    start_idx = max(args.initial_timestamp, 0)
    if args.end_timestamp is not None:
        end_idx = min(args.end_timestamp, T)
    else:
        end_idx = T
    if start_idx >= end_idx:
        raise ValueError(f"Empty time range: start={start_idx}, end={end_idx}, T={T}")

    if args.frame == "base":
        actual_lin = np.einsum("nij,nj->ni", np.transpose(R_B, (0, 2, 1)), pdot_B)
        desired_lin = lin_vel_d
        actual_ang = varphi_B
        desired_ang = ang_vel_d
    else:
        actual_lin = pdot_B
        desired_lin = np.einsum("nij,nj->ni", R_B, lin_vel_d)
        actual_ang = omega_B
        desired_ang = np.einsum("nij,nj->ni", R_B, ang_vel_d)

    plot_items = [
        ("linear velocity x", "m/s", desired_lin[:, 0], actual_lin[:, 0]),
        ("linear velocity y", "m/s", desired_lin[:, 1], actual_lin[:, 1]),
        ("yaw angular velocity", "rad/s", desired_ang[:, 2], actual_ang[:, 2]),
    ]

    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(3, 1, height_ratios=[3, 3, 3])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, unit, desired, actual) in enumerate(plot_items):
        desired_plot = desired[start_idx:end_idx]
        actual_plot = actual[start_idx:end_idx]
        rmse = np.sqrt(np.mean((desired_plot - actual_plot) ** 2))

        axes[i].plot(
            t[start_idx:end_idx],
            desired_plot,
            linestyle="--",
            color=color_cycle[0],
            label="desired",
        )
        axes[i].plot(
            t[start_idx:end_idx],
            actual_plot,
            linestyle="-",
            color=color_cycle[1],
            label="actual",
        )
        axes[i].grid(True)
        axes[i].set_ylabel(f"{name}\n[{unit}]")
        axes[i].set_title(f"RMSE: {rmse:.4f} {unit}", loc="right", fontsize=10)
        axes[i].legend(ncol=2, fontsize=9, loc="best")

    for ax in axes:
        ax.set_xlabel("Time step")

    fig.suptitle(f"Velocity Tracking Result ({args.frame} frame)", fontsize=13)
    plt.tight_layout()

    save_path = assets_dir / "velocity_tracking_result.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
