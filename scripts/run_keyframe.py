from swm.keyframe.extraction import extract_frames_from_video, extract_keyframes_from_frames
from pathlib import Path

"""
smooth_k是平滑窗口大小，用于减少镜头晃动和灯光闪烁引起的噪声，过大会导致微弱变化可能被忽略。
merge_pct是合并阈值，相邻极值点的能量变化＜merge_pct的会被合并。、
"""
if __name__ == "__main__":
    task_domain = "test"
    root_dir = Path(__file__).parent.parent
    videos_root = root_dir / "dataset" /  "videos" / task_domain
    frames_root = root_dir / "dataset" /  "frames" / task_domain
    keyframes_root = root_dir / "dataset" /  "keyframes" / task_domain
    extract_frames_from_video(videos_root, frames_root, max_workers=1)
    extract_keyframes_from_frames(frames_root, keyframes_root, smooth_k=5, merge_pct=0.5, max_workers=1, plot_energy=True)