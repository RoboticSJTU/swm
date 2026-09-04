"""Temporal-Gradient Keyframe Extraction."""

import shutil
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_frames(video_path: Path, frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    if any(frames_dir.glob("*.png")):
        return
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(video_path), str(frames_dir / "%04d.png")],
        check=True,
    )
    print(f"[frames] {video_path.name} -> {frames_dir}")


def save_unidomain_energy(frames_dir, save_path) -> None:
    """Save the pixel-space image energy used by UniDomain."""
    frame_paths = sorted(
        Path(frames_dir).glob("*.png"), key=lambda path: int(path.stem)
    )
    energies = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            gray = np.asarray(image.convert("L"), dtype=np.float64)
        energies.append(np.sum(gray * gray))
    np.save(save_path, np.asarray(energies, dtype=np.float64))


def save_temporal_gradient_energy(frames_dir, save_path) -> None:
    frame_paths = sorted(
        Path(frames_dir).glob("*.png"), key=lambda path: int(path.stem)
    )
    print("Calculating temporal gradient energy ...")
    if len(frame_paths) < 2:
        np.save(save_path, np.zeros(len(frame_paths), dtype=np.float64))
        return

    def load_gray(path):
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.float32)

    previous = current = load_gray(frame_paths[0])
    following = load_gray(frame_paths[1])
    energy = np.empty(len(frame_paths), dtype=np.float64)
    for center in range(len(frame_paths)):
        gradient = 0.5 * (following - previous)
        energy[center] = np.sum(gradient * gradient, dtype=np.float64)
        previous, current = current, following
        if center + 2 < len(frame_paths):
            following = load_gray(frame_paths[center + 2])
    np.save(save_path, energy)


def extract_temporal_gradient_keyframes(
    frames_dir: Path,
    keyframe_dir: Path,
    radius: int,
    smooth_k: int = 5,
    merge_pct: float = 0.5,
    plot_energy: bool = True,
) -> None:
    if any(keyframe_dir.glob("seg_*/*.png")):
        return

    images = sorted(frames_dir.glob("*.png"), key=lambda path: int(path.stem))
    if not images:
        raise ValueError(f"No frames found in {frames_dir}")

    keyframe_dir.mkdir(parents=True, exist_ok=True)
    energy_path = frames_dir.parent / "energies" / f"{frames_dir.name}_energies.npy"
    energy_path.parent.mkdir(parents=True, exist_ok=True)
    save_temporal_gradient_energy(frames_dir, energy_path)
    energy = np.load(energy_path)
    smooth_k = smooth_k | 1
    pad = smooth_k // 2
    energy = np.convolve(
        np.pad(energy, (pad, pad), mode="edge"),
        np.ones(smooth_k) / smooth_k,
        mode="valid",
    )

    frame_count = len(images)
    extrema = []
    for center in range(frame_count):
        left = max(center - radius, 0)
        right = min(center + radius, frame_count - 1)
        window = energy[left : right + 1]
        offset = center - left
        if offset == window.argmax() or offset == window.argmin():
            extrema.append(center)

    median = float(np.median(energy))
    merged = []
    for current in extrema:
        if merged:
            previous = merged[-1]
            current_energy = float(energy[current])
            previous_energy = float(energy[previous])
            scale = max(abs(current_energy), abs(previous_energy), 1e-9)
            if abs(current_energy - previous_energy) / scale <= merge_pct:
                if abs(current_energy - median) > abs(previous_energy - median):
                    merged[-1] = current
                continue
        merged.append(current)
    extrema = merged

    labels = ["peak"]
    if len(extrema) > 1:
        labels = []
        for index, current in enumerate(extrema):
            left = extrema[index - 1] if index else extrema[1]
            right = extrema[index + 1] if index < len(extrema) - 1 else extrema[-2]
            labels.append(
                "peak"
                if energy[current] >= energy[left] and energy[current] >= energy[right]
                else "valley"
            )

    first_peak_index = labels.index("peak")
    start = first_peak_index
    seen_valley = False
    segments = []
    for end in range(start + 1, len(extrema)):
        seen_valley = seen_valley or labels[end] == "valley"
        if labels[end] == "peak" and seen_valley:
            segments.append(extrema[start : end + 1])
            start = end
            seen_valley = False

    if segments and first_peak_index:
        segments[0] = sorted(set(extrema[:first_peak_index] + segments[0]))
    if not segments:
        segments = [list(extrema)]

    if len(extrema) >= 5:
        first_extrema, last_extrema = extrema[0], extrema[-1]
        extrema = extrema[1:-1]
        segments[0] = [frame for frame in segments[0] if frame != first_extrema]
        segments[-1] = [frame for frame in segments[-1] if frame != last_extrema]

    segments[0] = sorted({0, *segments[0]})
    segments[-1] = sorted({*segments[-1], frame_count - 1})

    if plot_energy:
        figure, axis = plt.subplots(figsize=(10, 3))
        axis.plot(energy, linewidth=1)
        axis.scatter(extrema, energy[extrema], color="red", s=25, zorder=3)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_linewidth(1.0)
        axis.tick_params(
            axis="both",
            which="both",
            direction="out",
            length=4,
            width=1.0,
            labelsize=10,
            top=False,
            right=False,
        )
        axis.grid(False)
        figure.tight_layout()
        figure.savefig(keyframe_dir / "energy_curve.png", dpi=300, bbox_inches="tight")
        plt.close(figure)

    for index, segment in enumerate(segments):
        segment_dir = keyframe_dir / f"seg_{index:02d}"
        segment_dir.mkdir(parents=True, exist_ok=True)
        for frame in segment:
            shutil.copy2(images[frame], segment_dir / images[frame].name)

    print(
        f"[temporal-gradient keyframes] {frames_dir.name}: "
        f"extrema={len(extrema)}, segments={len(segments)}, window_step={radius}"
    )
