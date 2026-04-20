from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageOps

IMG_DIR = Path("/inspire/hdd/project/robot-decision/xiaoyunxiao-240108120113/swm/tasks/images/swm")
LANDSCAPE_TARGET = (1280, 720)
PORTRAIT_TARGET = (720, 1280)
WORKERS = 16

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def resize_one(img_path: Path):
    try:
        with Image.open(img_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            w, h = img.size
            new_size = None

            # 横图：宽 > 高，且宽高都超过阈值时，缩放到 (1280, 720)
            if w > h and w > 1280 and h > 720:
                new_size = LANDSCAPE_TARGET

            # 竖图：宽 < 高，且宽高都超过阈值时，缩放到 (720, 1280)
            elif w < h and w > 720 and h > 1280:
                new_size = PORTRAIT_TARGET

            # 其他情况不改
            if new_size is not None:
                img = img.resize(new_size, Image.Resampling.LANCZOS)

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