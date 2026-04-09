from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageOps

IMG_DIR = Path("/home/xyx/下载/swm/tasks/images/swm_100")
TARGET_SIZE = (1280, 720)
WORKERS = 16

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def resize_one(img_path: Path):
    try:
        with Image.open(img_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            # 直接缩放到 1280x720
            img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

            save_kwargs = {}
            if img_path.suffix.lower() in {".jpg", ".jpeg"}:
                save_kwargs["quality"] = 95

            img.save(img_path, **save_kwargs)
        return True, img_path
    except Exception as e:
        return False, f"{img_path}: {e}"


def main():
    image_paths = [p for p in IMG_DIR.rglob("*") if p.suffix.lower() in EXTS]
    print(f"共找到 {len(image_paths)} 张图像")

    ok = 0
    bad = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(resize_one, p) for p in image_paths]
        for fut in as_completed(futures):
            success, info = fut.result()
            if success:
                ok += 1
            else:
                bad += 1
                print("失败：", info)

    print(f"完成：成功 {ok}，失败 {bad}")


if __name__ == "__main__":
    main()