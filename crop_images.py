from PIL import Image
import glob, os

for path in glob.glob("input_images/*.png"):
    with Image.open(path) as img:
        w, h = img.size
        if w >= 1024 and h >= 512:
            left = (w - 1024) // 2
            top = (h - 512) // 2
            cropped = img.crop((left, top, left + 1024, top + 512))
            cropped.save(f"assets/cover_images/{os.path.basename(path)}", format="PNG")
