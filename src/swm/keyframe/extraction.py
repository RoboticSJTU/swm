import shutil
import subprocess
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swm.keyframe.tools import energy_keyframes, load_energy, save_image_energy


def extract_frames(video_path: Path, frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    if list(frames_dir.glob("*.png")):
        return

    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(video_path), str(frames_dir / "%04d.png")],
        check=True,
    )
    print(f"[frames] {video_path.name} -> {frames_dir}")


def extract_keyframes(
    frames_dir: Path,
    keyframe_dir: Path,
    smooth_k: int = 5,
    merge_pct: float = 0.5,
    plot_energy: bool = True,
) -> None:
    if list(keyframe_dir.glob("seg_*/*.png")):
        return

    images = sorted(frames_dir.glob("*.png"), key=lambda path: int(path.stem))
    if not images:
        raise ValueError(f"No frames found in {frames_dir}")

    keyframe_dir.mkdir(parents=True, exist_ok=True)
    energy_path = frames_dir.parent / "energies" / f"{frames_dir.name}_energies.npy"
    energy_path.parent.mkdir(parents=True, exist_ok=True)
    if not energy_path.is_file():
        save_image_energy(frames_dir, energy_path)

    energy = load_energy(energy_path)
    if len(energy) != len(images):
        raise ValueError(
            f"Energy count {len(energy)} does not match frame count {len(images)}"
        )
    smooth_k = smooth_k | 1
    pad = smooth_k // 2
    energy = np.convolve(
        np.pad(energy, (pad, pad), mode="edge"),
        np.ones(smooth_k) / smooth_k,
        mode="valid",
    )

    frame_count = len(images)
    window_step = 10 + 10 * max(0, (frame_count - 1) // 500)
    window_step = min(90, window_step)
    extrema = sorted(set(map(int, energy_keyframes(energy, window_step))))

    median = float(np.median(energy))
    merged = []
    for current in extrema:
        if not merged:
            merged.append(current)
            continue

        previous = merged[-1]
        difference = abs(float(energy[current]) - float(energy[previous]))
        scale = max(abs(float(energy[current])), abs(float(energy[previous])), 1e-9)
        if difference / scale <= merge_pct:
            if abs(float(energy[current]) - median) > abs(float(energy[previous]) - median):
                merged[-1] = current
        else:
            merged.append(current)
    extrema = merged

    labels = []
    for index, current in enumerate(extrema):
        if len(extrema) == 1:
            labels = ["peak"]
            break
        left = extrema[index - 1] if index > 0 else extrema[index + 1]
        right = extrema[index + 1] if index < len(extrema) - 1 else extrema[index - 1]
        labels.append("peak" if energy[current] >= energy[left] and energy[current] >= energy[right] else "valley")

    segments = []
    first_peak_index = next((i for i, label in enumerate(labels) if label == "peak"), None)
    index = first_peak_index
    while index is not None:
        seen_valley = False
        for end in range(index + 1, len(extrema)):
            seen_valley = seen_valley or labels[end] == "valley"
            if labels[end] == "peak" and seen_valley:
                segments.append(extrema[index:end + 1])
                index = end
                break
        else:
            break

    if segments and first_peak_index:
        segments[0] = sorted(set(extrema[:first_peak_index] + segments[0]))
    if not segments:
        segments = [list(extrema)]

    if len(extrema) >= 5:
        first_extrema, last_extrema = extrema[0], extrema[-1]
        extrema = extrema[1:-1]
        segments[0] = [frame for frame in segments[0] if frame != first_extrema]
        segments[-1] = [frame for frame in segments[-1] if frame != last_extrema]

    segments[0] = sorted(set([0] + segments[0]))
    segments[-1] = sorted(set(segments[-1] + [frame_count - 1]))

    if plot_energy:
        figure, axis = plt.subplots(figsize=(10, 3))
        axis.plot(energy, linewidth=1)
        axis.scatter(extrema, energy[extrema], color="red", s=25, zorder=3)
        axis.set_title("")
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(1.0)
        axis.spines["bottom"].set_linewidth(1.0)
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

    print(f"[keyframes] {frames_dir.name}: extrema={len(extrema)}, segments={len(segments)}, window_step={window_step}")
