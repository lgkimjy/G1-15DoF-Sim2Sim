#!/usr/bin/env python3

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


TORSO_JOINT_INDICES = [12, 13, 14]


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


def reshape_rot_mat(data):
    if data.ndim == 1:
        data = data.reshape(-1, 9)
    return data.reshape((-1, 3, 3)).transpose(0, 2, 1)


def rot_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.zeros((angle.shape[0], 3, 3))
    R[:, 0, 0] = 1.0
    R[:, 1, 1] = c
    R[:, 1, 2] = -s
    R[:, 2, 1] = s
    R[:, 2, 2] = c
    return R


def rot_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.zeros((angle.shape[0], 3, 3))
    R[:, 0, 0] = c
    R[:, 0, 2] = s
    R[:, 1, 1] = 1.0
    R[:, 2, 0] = -s
    R[:, 2, 2] = c
    return R


def rot_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.zeros((angle.shape[0], 3, 3))
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R


def torso_relative_rotation(torso_jpos):
    R_roll = rot_x(torso_jpos[:, 0])
    R_pitch = rot_y(torso_jpos[:, 1])
    R_yaw = rot_z(torso_jpos[:, 2])
    return np.matmul(np.matmul(R_roll, R_pitch), R_yaw)


def rotation_to_rpy(R):
    pitch_arg = np.clip(-R[:, 2, 0], -1.0, 1.0)
    roll = np.arctan2(R[:, 2, 1], R[:, 2, 2])
    pitch = np.arcsin(pitch_arg)
    yaw = np.arctan2(R[:, 1, 0], R[:, 0, 0])
    return np.unwrap(np.column_stack((roll, pitch, yaw)), axis=0)


def main():
    parser = argparse.ArgumentParser(description="Plot torso orientation change during walking")
    parser.add_argument("directory", type=str, help="Directory containing stateData.h5 file")
    parser.add_argument("-i", "--initial_timestamp", type=int, default=0, help="Initial timestamp")
    parser.add_argument("-e", "--end_timestamp", type=int, default=None, help="End timestamp")
    parser.add_argument(
        "--frame",
        choices=["world", "pelvis"],
        default="world",
        help="world: pelvis orientation + torso joints, pelvis: torso joints only",
    )
    parser.add_argument("--unit", choices=["deg", "rad"], default="deg", help="Angle unit")
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
            R_B = reshape_rot_mat(read_h5_dataset(f, "fbk/R_B"))
            if "fbk/jpos" in f:
                jpos = read_h5_dataset(f, "fbk/jpos")
            else:
                qpos = read_h5_dataset(f, "fbk/qpos")
                jpos = qpos[:, 7:]
    except OSError as e:
        raise SystemExit(
            f"Failed to open {log_file_path}: {e}\n"
            "This usually means the HDF5 log was not closed/flushed cleanly. "
            "Close the simulator window normally and use the newly created log."
        )

    if jpos.ndim == 1:
        jpos = jpos.reshape(-1, 15)

    T = min(R_B.shape[0], jpos.shape[0])
    R_B = R_B[:T]
    jpos = jpos[:T]
    t = np.arange(T)

    start_idx = max(args.initial_timestamp, 0)
    if args.end_timestamp is not None:
        end_idx = min(args.end_timestamp, T)
    else:
        end_idx = T
    if start_idx >= end_idx:
        raise ValueError(f"Empty time range: start={start_idx}, end={end_idx}, T={T}")

    torso_jpos = jpos[:, TORSO_JOINT_INDICES]
    R_torso_pelvis = torso_relative_rotation(torso_jpos)

    if args.frame == "world":
        R_torso = np.matmul(R_B, R_torso_pelvis)
    else:
        R_torso = R_torso_pelvis

    R_torso = R_torso[start_idx:end_idx]
    torso_jpos = torso_jpos[start_idx:end_idx]
    t_plot = t[start_idx:end_idx]

    rpy = rotation_to_rpy(R_torso)
    rpy_delta = rpy - rpy[0]

    scale = 180.0 / np.pi if args.unit == "deg" else 1.0
    unit_label = "deg" if args.unit == "deg" else "rad"
    rpy_delta = rpy_delta * scale
    torso_jpos = torso_jpos * scale

    plot_items = [
        (rpy_delta[:, 0], "delta roll"),
        (rpy_delta[:, 1], "delta pitch"),
        (rpy_delta[:, 2], "delta yaw"),
    ]

    fig = plt.figure(figsize=(10, 7))
    gs = GridSpec(3, 1, height_ratios=[3, 3, 3])
    axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (data, label) in enumerate(plot_items):
        peak = np.max(np.abs(data))
        axes[i].plot(
            t_plot,
            data,
            linestyle="-",
            color=color_cycle[i % len(color_cycle)],
            label=label,
        )
        axes[i].grid(True)
        axes[i].set_ylabel(f"{label}\n[{unit_label}]")
        axes[i].set_title(f"peak abs: {peak:.3f} {unit_label}", loc="right", fontsize=10)
        axes[i].legend(fontsize=9, loc="best")

    axes[0].text(
        0.01,
        0.95,
        "torso joint range [roll, pitch, yaw]: "
        f"[{np.ptp(torso_jpos[:, 0]):.3f}, "
        f"{np.ptp(torso_jpos[:, 1]):.3f}, "
        f"{np.ptp(torso_jpos[:, 2]):.3f}] {unit_label}",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )

    for ax in axes:
        ax.set_xlabel("Time step")

    fig.suptitle(f"Torso Orientation Change ({args.frame} frame)", fontsize=13)
    plt.tight_layout()

    save_path = assets_dir / "torso_orientation_delta.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot: {save_path}")
    plt.show()


if __name__ == "__main__":
    main()
