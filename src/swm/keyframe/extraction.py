from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from swm.keyframe.tools import save_image_energy, load_energy, energy_keyframes
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_frames_from_video(videos_root, frames_root, max_workers = 1) -> None:
    futures = {}
    frames_root.mkdir(parents=True, exist_ok=True) 
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for mp4 in sorted(videos_root.glob("*.mp4")):
            out_dir = frames_root / mp4.stem
            out_dir.mkdir(parents=True, exist_ok=True)

            # 已抽过：存在至少一张 png 就跳过
            if any(out_dir.glob("*.png")):
                print(f"[skip] {mp4.name} (already has png)")
                continue

            command = ['ffmpeg', '-loglevel', 'error', '-i', str(mp4), str(out_dir / "%04d.png")]
            futures[ex.submit(subprocess.run, command, check=True)] = (mp4, out_dir)

        for fut in as_completed(futures):
            mp4, out_dir = futures[fut]
            fut.result()
            print(f"Extract frames from {mp4} successfully!")

def extract_keyframes_from_frames(frames_root, keyframes_root, smooth_k=5, merge_pct=0.5, max_workers = 16, plot_energy=True) -> None:
    
    keyframes_root.mkdir(parents=True, exist_ok=True)
    eps = sorted([p for p in frames_root.iterdir() if p.is_dir() and p.name != "energies"], key=lambda p: p.name)
    plot_lock = threading.Lock()
    def _run_one(ep: Path):
        imgs = sorted(ep.glob("*.png"), key=lambda p: int(p.stem))
        if not imgs:
            return

        n = len(imgs)
        # # 人拍。由于镜头晃，所以窗口小一些。
        # window_step = 10 + 10 * max(0, (n - 1) // 500)

        # droid
        # window_step = 10 + 10 * max(0, (n - 1) // 200)

        # agibot
        window_step = 20 + 20 * max(0, (n - 1) // 500)


        window_step = min(90, window_step)
        
        out = keyframes_root / ep.name
        out.mkdir(parents=True, exist_ok=True)
        
        # 已抽过：存在至少一张 png 就跳过
        if next(out.rglob("*.png"), None) is not None:
            return
        # 1) energy
        energies_dir = frames_root / "energies"
        energies_dir.mkdir(parents=True, exist_ok=True)
        energies_path = energies_dir / f"{ep.name}_energies.npy"

        if not energies_path.exists():
            save_image_energy(ep, energies_path)
        e = load_energy(energies_path)

        # 2) smooth
        k = smooth_k | 1
        pad = k // 2
        e = np.convolve(np.pad(e, (pad, pad), mode="edge"), np.ones(k) / k, mode="valid")

        # 3) extrema
        ext = sorted(set(map(int, energy_keyframes(e, window_step))))

        # 4) merge close/noisy extrema
        med = float(np.median(e))
        merged = []
        for t in ext:
            if not merged:
                merged.append(t)
                continue
            p = merged[-1]
            pct = abs(float(e[t]) - float(e[p])) / max(abs(float(e[t])), abs(float(e[p])), 1e-9)
            if pct <= merge_pct:
                if abs(float(e[t]) - med) > abs(float(e[p]) - med):
                    merged[-1] = t
            else:
                merged.append(t)
        ext = merged

        # 5) peak/valley label (neighbor compare)
        labels = []
        for i, t in enumerate(ext):
            if len(ext) == 1:
                labels = ["peak"]
                break
            l = ext[i - 1] if i > 0 else ext[i + 1]
            r = ext[i + 1] if i < len(ext) - 1 else ext[i - 1]
            labels.append("peak" if (e[t] >= e[l] and e[t] >= e[r]) else "valley")

        # 6) segments: peak ... valley ... peak (share boundary peak)
        segs = []
        first_peak_i = next((i for i, lab in enumerate(labels) if lab == "peak"), None)
        i = first_peak_i
        while i is not None:
            seen_valley = False
            for j in range(i + 1, len(ext)):
                seen_valley |= (labels[j] == "valley")
                if labels[j] == "peak" and seen_valley:
                    segs.append(ext[i : j + 1])
                    i = j
                    break
            else:
                break

        # 把第一个 peak 之前的 extrema 全部并入第一组
        if segs and first_peak_i and first_peak_i > 0:
            segs[0] = sorted(set(ext[:first_peak_i] + segs[0]))

        # 分组兜底：如果分不出peak ... valley ... peak，则把所有 extrema 都并入第一组
        if not segs:
            segs = [list(ext)]

        # 如果总极值点个数多，则删掉首尾极值点，减少冗余。
        if len(ext) >= 5:
            drop_first, drop_last = ext[0], ext[-1]
            ext = ext[1:-1]  
            segs[0]  = [t for t in segs[0]  if t != drop_first]
            segs[-1] = [t for t in segs[-1] if t != drop_last]
        
        # 加入全局首尾帧
        first, last = 0, n - 1
        segs[0]  = sorted(set([first] + segs[0]))
        segs[-1] = sorted(set(segs[-1] + [last]))

        # 7) plot curve + extrema points
        if plot_energy: 
            with plot_lock:
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(e, linewidth=1)
                ax.scatter(ext, e[ext], color="red", s=25, zorder=3)
                ax.set(title=f"Energy diff Curve: {ep.name}", xlabel="Frame Index", ylabel="Energy")
                fig.tight_layout()
                fig.savefig(out / "energy_curve.png")
                plt.close(fig)

        # 8) write folders
        for si, seg in enumerate(segs):
            seg_dir = out / f"seg_{si:02d}"
            seg_dir.mkdir(parents=True, exist_ok=True)
            for t in seg:
                if 0 <= t < n:
                    shutil.copy2(imgs[t], seg_dir / imgs[t].name)

        print(f"[done] {ep.name}: extrema={len(ext)}, segments={len(segs)}, window_step={window_step}, smooth_k={smooth_k}, merge_pct={merge_pct}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_run_one, ep) for ep in eps]
        for fut in as_completed(futs):
            fut.result()
            
    print(f"任务完成：输出在 {keyframes_root}")
