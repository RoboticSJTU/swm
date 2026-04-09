from pathlib import Path
import numpy as np
from numpy.typing import NDArray
import sys
import subprocess
from PIL import Image, ImageDraw
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, wait

def save_image_energy_仅用能量(frames_dir, save_path) -> None:
    """Precalculate and save the image energy of frames in the given directory.
       Frames in the directory should be named as "00001.png", "00002.png", etc.
       The energies are calculated by the sum of square of pixel values of the original image.

    Args:
        frames_dir (Paths): The directory containing frames.
        save_path (Paths): The path to save energies.

    Returns:
        None
    """

    frames_dir = Path(frames_dir)
    frame_path_list = sorted(list(frames_dir.iterdir()))

    print("Calculating energies in pixel space ...")
    energies = []
    for frame_path in frame_path_list:
        gray = Image.open(frame_path).convert("L")
        energy = np.sum(np.array(gray, dtype=np.float64) ** 2) # energy of image f = \sum_{x,y} |f(x,y)|^2
        energies.append(energy)

    np.save(save_path, energies)
    print(f"Save energies to {save_path} successfully!")

def save_image_energy(frames_dir, save_path) -> None:
    frames_dir = Path(frames_dir)
    frame_path_list = sorted(list(frames_dir.iterdir()))

    print("Calculating frame-diff energies ...")
    energies = []

    prev = None
    for frame_path in frame_path_list:
        gray = Image.open(frame_path).convert("L")
        cur = np.array(gray, dtype=np.float64)

        if prev is None:
            energy = 0.0
        else:
            diff = cur - prev
            energy = np.sum(diff ** 2)   # frame-diff energy: sum((I_t - I_{t-1})^2)

        energies.append(energy)
        prev = cur

    np.save(save_path, np.array(energies, dtype=np.float64))

def load_energy(save_path) -> NDArray:
    """Load energies from the given path.

    Args:
        save_path (Paths): The path to load energies.

    Returns:
        NDArray: 2D array, the energy array.
    """

    return np.load(save_path)

def energy_keyframes(energies: NDArray, delta: int = 50) -> NDArray:
    """Select keyframes based on the energies of frame sequences.
       The keyframes are selected by the extreme point of the energies.
       
    Args:
        energies (NDArray): The energies of frames.
        delta (int): The range of local neighborhood. Defaults to 50.

    Returns:
        NDArray: 1D array, the index of keyframes.
    """

    total_frames = len(energies)
    keyframes_index = []   # the first keyframe is the first frame

    for center in range(total_frames):
        # local neighborhood boundary
        left = max(center - delta, 0)
        right = min(center + delta, total_frames - 1)
        
        window = energies[left:right + 1]  
        if window.argmax() == center - left:
                    keyframes_index.append(center)
        if window.argmin() == center - left:
                    keyframes_index.append(center)

    return np.array(keyframes_index)

def frames_crop(
    frames_root,
    frames_crop_root,
    model_path="/inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/roboDM/llmdet_large",
    text_labels=["robot hand"],
    box_th=0.28,
    sample_stride=30,
    margin_frac=0.15,
    batch_size=1,
) -> None:

    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    frames_root = Path(frames_root)
    frames_crop_root = Path(frames_crop_root)
    frames_crop_root.mkdir(parents=True, exist_ok=True)
    # vis_root = frames_crop_root.parent / f"{frames_crop_root.name}_vis"
    model_path = Path(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    ).to(device).eval()

    eps = sorted([p for p in frames_root.iterdir() if p.is_dir() and p.name != "energies"],key=lambda p: p.name)
    
    # NEW: ffmpeg 并发池
    ffmpeg_ex = ThreadPoolExecutor(max_workers=100)
    pending_ffmpeg = []

    try:
        for ep in tqdm(eps, desc="episodes"):
            imgs = sorted(ep.glob("*.png"))
            if not imgs:
                continue

            out_ep = frames_crop_root / ep.name
            out_ep.mkdir(parents=True, exist_ok=True)
            if any(out_ep.glob("*.png")):
                continue

            # vis_ep = vis_root / ep.name
            # vis_ep.mkdir(parents=True, exist_ok=True)

            stem = imgs[0].stem
            if not stem.isdigit():
                print(f"[skip] {ep.name}: png name not numeric, e.g. {imgs[0].name}")
                continue
            width = len(stem)
            start_number = int(stem)
            pattern = f"%0{width}d.png"

            n = len(imgs)
            idx = np.arange(0, n, max(1, int(sample_stride)), dtype=int)
            if idx.size == 0 or idx[-1] != n - 1:
                idx = np.append(idx, n - 1)
            sample_paths = [imgs[i] for i in np.unique(idx)]

            with Image.open(sample_paths[0]) as im0:
                W, H = im0.size

            boxes_all = []
            for i in range(0, len(sample_paths), batch_size):
                batch_paths = sample_paths[i:i + batch_size]

                batch_imgs = []
                for p in batch_paths:
                    with Image.open(p) as im:
                        batch_imgs.append(im.convert("RGB"))

                inputs = processor(
                    images=batch_imgs,
                    text=[list(text_labels)] * len(batch_imgs),
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                results = processor.post_process_grounded_object_detection(
                    outputs,
                    threshold=float(box_th),
                    target_sizes=[(H, W)] * len(batch_imgs),
                )

                for p, im, r in zip(batch_paths, batch_imgs, results):
                    boxes = r["boxes"]
                    scores = r["scores"]

                    if boxes is not None and len(boxes) > 0:
                        boxes = boxes.detach().cpu().numpy()
                        scores = scores.detach().cpu().numpy()
                        for b, s in zip(boxes, scores):
                            x1b, y1b, x2b, y2b = map(float, b.tolist())
                            boxes_all.append([x1b, y1b, x2b, y2b])

                    # im_vis = im.copy()
                    # draw = ImageDraw.Draw(im_vis)

                    # if boxes is not None and len(boxes) > 0:
                    #     boxes = boxes.detach().cpu().numpy()
                    #     scores = scores.detach().cpu().numpy()
                    #     for b, s in zip(boxes, scores):
                    #         x1b, y1b, x2b, y2b = map(float, b.tolist())
                    #         draw.rectangle([x1b, y1b, x2b, y2b], outline="red", width=3)
                    #         draw.text((x1b, max(0, y1b - 12)), f"{float(s):.2f}", fill="red")
                    #         boxes_all.append([x1b, y1b, x2b, y2b])

                    # im_vis.save(vis_ep / p.name)

            if not boxes_all:
                x1, y1, x2, y2 = 0.0, 0.0, float(W), float(H)
            else:
                arr = np.asarray(boxes_all, dtype=np.float32)
                x1 = float(np.quantile(arr[:, 0], 0.05))
                y1 = float(np.quantile(arr[:, 1], 0.05))
                x2 = float(np.quantile(arr[:, 2], 0.95))
                y2 = float(np.quantile(arr[:, 3], 0.95))

                mw = (x2 - x1) * float(margin_frac)
                mh = (y2 - y1) * float(margin_frac)
                x1, y1 = max(0.0, x1 - mw), max(0.0, y1 - mh)
                x2, y2 = min(float(W), x2 + mw), min(float(H), y2 + mh)
                x2, y2 = max(x1 + 2.0, x2), max(y1 + 2.0, y2)

            w = int(round(x2 - x1))
            h = int(round(y2 - y1))
            x = int(round(x1))
            y = int(round(y1))

            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-start_number", str(start_number),
                "-i", str(ep / pattern),
                "-vf", f"crop={w}:{h}:{x}:{y}",
                "-vsync", "0",
                str(out_ep / pattern),
            ]

            # 不阻塞主线程，提交给线程池并发跑
            pending_ffmpeg.append(ffmpeg_ex.submit(subprocess.run, cmd, check=True))

        # NEW: 函数返回前等待全部 ffmpeg 完成，并抛出任何失败
        if pending_ffmpeg:
            for f in pending_ffmpeg:
                f.result()

    finally:
        ffmpeg_ex.shutdown(wait=True)
