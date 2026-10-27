import json
import math
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]


def extract_keyframes_embedding(
    frames_root,
    keyframes_root,
    smooth_k=5,
    merge_pct=0.5,
    batch_size=128,
    num_workers=8,
    post_workers=8,
    plot_curve=False,
    use_hardlink=True,
):
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from scipy.signal import find_peaks
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoModel, AutoProcessor

    class FrameDataset(Dataset):
        def __init__(self, img_paths, model_path):
            self.img_paths = [str(p) for p in img_paths]
            self.model_path = model_path
            self.processor = None

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx):
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    use_fast=True,
                )

            with Image.open(self.img_paths[idx]) as img:
                return self.processor(
                    images=img.convert("RGB"),
                    return_tensors="pt",
                )["pixel_values"][0]

    def process_episode(ep_name, imgs, emb_path, diff_path, out, smooth_k, merge_pct, plot_curve, use_hardlink, window_step):
        imgs = [Path(p) for p in imgs]
        out = Path(out)
        emb = np.load(emb_path, mmap_mode="r")
        n = len(imgs)

        if Path(diff_path).exists():
            e = np.load(diff_path)
        else:
            e = np.zeros(n, dtype=np.float32)
            if n >= 2:
                sim = np.sum(emb[1:] * emb[:-1], axis=1)
                e[1:] = 1.0 - np.clip(sim, -1.0, 1.0)
            np.save(diff_path, e)

        k = smooth_k | 1
        pad = k // 2
        e = np.convolve(np.pad(e, (pad, pad), mode="edge"), np.ones(k) / k, mode="valid")

        peaks, _ = find_peaks(e, distance=max(1, window_step))
        valleys, _ = find_peaks(-e, distance=max(1, window_step))
        ext = sorted(set(peaks.tolist() + valleys.tolist()))

        med = float(np.median(e))
        keep = []
        for t in ext:
            if not keep:
                keep.append(t)
                continue
            p = keep[-1]
            pct = abs(float(e[t]) - float(e[p])) / max(abs(float(e[t])), abs(float(e[p])), 1e-9)
            if pct <= merge_pct:
                if abs(float(e[t]) - med) > abs(float(e[p]) - med):
                    keep[-1] = t
            else:
                keep.append(t)
        ext = keep

        labels = []
        for i, t in enumerate(ext):
            if len(ext) == 1:
                labels = ["peak"]
                break
            l = ext[i - 1] if i > 0 else ext[i + 1]
            r = ext[i + 1] if i < len(ext) - 1 else ext[i - 1]
            labels.append("peak" if e[t] >= e[l] and e[t] >= e[r] else "valley")

        segs = []
        i = next((i for i, x in enumerate(labels) if x == "peak"), None)
        first_peak_i = i

        while i is not None:
            seen_valley = False
            for j in range(i + 1, len(ext)):
                seen_valley |= labels[j] == "valley"
                if labels[j] == "peak" and seen_valley:
                    segs.append(ext[i:j + 1])
                    i = j
                    break
            else:
                break

        if segs and first_peak_i is not None and first_peak_i > 0:
            segs[0] = sorted(set(ext[:first_peak_i] + segs[0]))
        if not segs:
            segs = [list(ext)]

        if len(ext) >= 5:
            a, b = ext[0], ext[-1]
            ext = ext[1:-1]
            segs[0] = [t for t in segs[0] if t != a]
            segs[-1] = [t for t in segs[-1] if t != b]

        segs[0] = sorted(set([0] + segs[0]))
        segs[-1] = sorted(set(segs[-1] + [n - 1]))

        if plot_curve:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(e, linewidth=1)
            if ext:
                ax.scatter(ext, e[ext], color="red", s=25, zorder=3)
            ax.set(
                title=f"Embedding diff Curve: {ep_name}",
                xlabel="Frame Index",
                ylabel="Cosine Distance",
            )
            fig.tight_layout()
            fig.savefig(out / "embedding_curve.png")
            plt.close(fig)

        for i, seg in enumerate(segs):
            seg_dir = out / f"seg_{i:02d}"
            seg_dir.mkdir(parents=True, exist_ok=True)
            for t in seg:
                if 0 <= t < n:
                    src = imgs[t]
                    dst = seg_dir / src.name
                    if dst.exists():
                        continue
                    if use_hardlink:
                        try:
                            os.link(src, dst)
                        except Exception:
                            shutil.copy2(src, dst)
                    else:
                        shutil.copy2(src, dst)

        (out / ".done").write_text("ok")
        return f"[done] {ep_name}: extrema={len(ext)}, segments={len(segs)}, window_step={window_step}"

    frames_root = Path(frames_root)
    keyframes_root = Path(keyframes_root)
    keyframes_root.mkdir(parents=True, exist_ok=True)

    model_path = ROOT_DIR / "siglip2-giant-opt-patch16-384"
    emb_dir = frames_root / "embeddings_siglip2"
    diff_dir = frames_root / "embedding_diffs_siglip2"
    emb_dir.mkdir(parents=True, exist_ok=True)
    diff_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model = AutoModel.from_pretrained(
        model_path,
        dtype=dtype,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device).eval()

    eps = sorted(
        [
            p for p in frames_root.iterdir()
            if p.is_dir() and p.name not in {"energies", "embeddings_siglip2", "embedding_diffs_siglip2"}
        ],
        key=lambda p: p.name,
    )

    pending = set()
    with ThreadPoolExecutor(max_workers=post_workers) as pool:
        for ep in eps:
            imgs = sorted(ep.glob("*.png"), key=lambda p: int(p.stem))
            if not imgs:
                continue

            n = len(imgs)
            window_step = 10 + 10 * ((n - 1) // 500)
            out = keyframes_root / ep.name
            out.mkdir(parents=True, exist_ok=True)

            if (out / ".done").exists():
                print(f"[skip] {ep.name}")
                continue

            emb_path = emb_dir / f"{ep.name}.npy"
            diff_path = diff_dir / f"{ep.name}_diffs_{window_step}.npy"

            if not emb_path.exists():
                loader = DataLoader(
                    FrameDataset(imgs, model_path),
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=device == "cuda",
                    persistent_workers=num_workers > 0,
                    prefetch_factor=4 if num_workers > 0 else None,
                )

                emb = None
                start = 0
                for pixel_values in loader:
                    pixel_values = pixel_values.to(device, non_blocking=True)
                    with torch.inference_mode():
                        feats = F.normalize(
                            model.get_image_features(pixel_values=pixel_values),
                            dim=-1,
                        )
                    feats = feats.float().cpu().numpy()

                    if emb is None:
                        emb = np.empty((n, feats.shape[1]), dtype=np.float32)

                    emb[start:start + len(feats)] = feats
                    start += len(feats)

                np.save(emb_path, emb)

            pending.add(pool.submit(
                process_episode,
                ep.name,
                [str(p) for p in imgs],
                str(emb_path),
                str(diff_path),
                str(out),
                smooth_k,
                merge_pct,
                plot_curve,
                use_hardlink,
                window_step,
            ))

            if len(pending) >= post_workers * 2:
                done = next(as_completed(pending))
                print(done.result())
                pending.remove(done)

        for done in as_completed(pending):
            print(done.result())

    print(f"任务完成：输出在 {keyframes_root}")

def extract_keyframes_uniform_sampling(
    src_root,
    dst_root,
    fps=None,
    group_size=3,
    *,
    source_fps=None,
    video_root=None,
):
    """Sample complete FFmpeg PNG sequences at a rate measured in frames/second.

    AgiBot defaults to one frame every four seconds; other datasets use one
    frame per second. Pass ``source_fps`` for a known PNG sequence rate, or
    ``video_root`` to probe each original video (including nested task folders).
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    fps = (0.25 if src_root.name.lower() == "agibot" else 1.0) if fps is None else float(fps)
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite sampling rate")
    if not isinstance(group_size, int) or group_size < 2:
        raise ValueError("group_size must be an integer >= 2")
    if source_fps is not None:
        source_fps = float(source_fps)
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError("source_fps must be a positive finite PNG sequence rate")
    if dst_root.exists() and any(dst_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {dst_root}")

    videos = {}
    if source_fps is None:
        video_root = Path(video_root) if video_root is not None else ROOT_DIR / "dataset" / "videos" / src_root.name
        for video in video_root.rglob("*.mp4"):
            videos.setdefault(video.stem, []).append(video)

    dst_root.mkdir(parents=True, exist_ok=True)

    task_dirs = sorted(
        [
            p for p in src_root.iterdir()
            if p.is_dir() and p.name.split("_")[-1].isdigit()
        ],
        key=lambda p: int(p.name.split("_")[-1])
    )

    for task_dir in task_dirs:
        frames = sorted(task_dir.glob("*.png"), key=lambda p: int(p.stem))
        if not frames:
            continue
        frame_rate = source_fps
        if frame_rate is None:
            matches = videos.get(task_dir.name, [])
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one source video for {task_dir.name} under {video_root}; "
                    "pass video_root or the actual source_fps explicitly"
                )
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=r_frame_rate,avg_frame_rate", "-of", "json", str(matches[0])],
                check=True, capture_output=True, text=True,
            )
            stream = json.loads(probe.stdout)["streams"][0]
            # The original FFmpeg PNG extraction uses this nominal output rate,
            # including duplicated frames in variable-frame-rate source videos.
            for field in ("r_frame_rate", "avg_frame_rate"):
                try:
                    frame_rate = float(Fraction(stream.get(field, "0/1")))
                except (ValueError, ZeroDivisionError):
                    continue
                if math.isfinite(frame_rate) and frame_rate > 0:
                    break
            else:
                raise ValueError(f"No valid frame rate found for {matches[0]}")

        if fps >= frame_rate:
            sampled = list(frames)
        else:
            count = math.floor((len(frames) - 1) * fps / frame_rate + 1e-9) + 1
            sampled = [frames[min(len(frames) - 1, math.floor(i * frame_rate / fps + 0.5))]
                       for i in range(count)]
        if sampled[-1] != frames[-1]:
            sampled.append(frames[-1])   # 强行补最后一帧

        task_out = dst_root / task_dir.name
        task_out.mkdir(parents=True, exist_ok=True)

        starts = list(range(0, max(len(sampled) - group_size + 1, 1), group_size - 1))
        last_start = max(len(sampled) - group_size, 0)
        if starts[-1] != last_start:
            starts.append(last_start)    # 强行补最后一组，让最后一帧落进去

        seg_id = 0
        for i in starts:
            seg_dir = task_out / f"seg_{seg_id:02d}"
            seg_dir.mkdir(parents=True, exist_ok=True)
            for frame in sampled[i:i + group_size]:
                shutil.copy2(frame, seg_dir / frame.name)
            seg_id += 1

        print(f"{task_dir.name}: total={len(frames)}, sampled={len(sampled)}, segs={seg_id}")
    print("done")


if __name__ == "__main__":
    # # uniform
    # dataset = "human"  # "agibot": every 4 seconds; other datasets: every second
    # fps = 0.25 if dataset == "agibot" else 1.0
    # group_size = 3
    # src_root = ROOT_DIR / "dataset" / "frames" / dataset
    # dst_root = ROOT_DIR / "dataset" / "keyframes" / f"{dataset}_uniform_sampling_fps_{fps}_group_{group_size}"
    # extract_keyframes_uniform_sampling(src_root, dst_root, fps, group_size)
    
    # embedding
    dataset_dir = ROOT_DIR / "dataset"
    extract_keyframes_embedding(
        frames_root=dataset_dir / "frames" / "human",
        keyframes_root=dataset_dir / "keyframes" / "human_embedding_siglip2_10window",
        smooth_k=5,
        merge_pct=0.5,
        batch_size=2000,
        num_workers=32,
        post_workers=32,
        plot_curve=True,
        use_hardlink=True,
    )
