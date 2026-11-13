import numpy as np
from PIL import Image

class Utils:

    def __init__(self):
        pass


    def _to_array(self, img) -> np.ndarray:
        if isinstance(img, np.ndarray):
            arr = img
        elif isinstance(img, Image.Image):
            arr = np.array(img.convert("RGB"))
        else:
            arr = np.array(Image.open(img).convert("RGB"))
        return arr.astype(np.float64)


    def psnr(self, cover_image, stego_image, max_value: float = 255.0) -> float:
        a = self._to_array(cover_image)
        b = self._to_array(stego_image)

        if a.shape != b.shape:
            raise ValueError(f"Tailles différentes : {a.shape} vs {b.shape}. ""Redimensionne la cover à la taille de la stego avant de comparer.")

        mse = np.mean((a - b) ** 2)
        if mse == 0:
            return float("inf")
        return float(10 * np.log10(max_value ** 2 / mse))