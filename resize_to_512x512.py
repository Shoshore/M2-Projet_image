from PIL import Image
import glob, os

for path in glob.glob("assets/images/*.pgm"):
    filename = os.path.splitext(os.path.basename(path))[0]
    with Image.open(path) as img:
        resized_img = img.resize((512, 512), Image.Resampling.LANCZOS)
        
        resized_img.save(f"assets/secret_images/{filename}.png", format="PNG")
