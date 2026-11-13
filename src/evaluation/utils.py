from numpy import mean, log10, array, float64, ndarray, unpackbits, uint8
from PIL.Image import Image, open
from skimage.metrics import structural_similarity

class Utils:

    def __init__(self):
        pass


    def _to_array(self, img) -> ndarray:
        if isinstance(img, ndarray):
            arr = img
        elif isinstance(img, Image):
            arr = array(img.convert("RGB"))
        else:
            arr = array(open(img).convert("RGB"))
        return arr.astype(float64)


    def psnr(self, cover_image, stego_image, max_value: float = 255.0) -> float:
        a = self._to_array(cover_image)
        b = self._to_array(stego_image)

        if a.shape != b.shape:
            raise ValueError(f"Tailles différentes : {a.shape} vs {b.shape}. ""Redimensionne la cover à la taille de la stego avant de comparer.")

        mse = mean((a - b) ** 2)
        if mse == 0:
            return float("inf")
        return float(10 * log10(max_value ** 2 / mse))

    
    def ssim(self, cover_image, stego_image, max_value: float = 255.0) -> float:
        a = self._to_array(cover_image)
        b = self._to_array(stego_image)

        if a.shape != b.shape:
            raise ValueError(f"Tailles différentes : {a.shape} vs {b.shape}. ""Redimensionne la cover à la taille de la stego avant de comparer.")

        return float(structural_similarity(
            a, b,
            data_range=max_value,
            channel_axis=2,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        ))


    def bit_accuracy(self, hidden_image, extracted_image):
        a = self._to_array(hidden_image).astype(uint8)
        b = self._to_array(extracted_image).astype(uint8)
        if a.shape != b.shape:
            raise ValueError(f"Tailles différentes : {a.shape} vs {b.shape}. " "Redimensionne l'image cachée à la taille de l'extraite avant de comparer.")
        bits_a = unpackbits(a)
        bits_b = unpackbits(b)
        return float(mean(bits_a == bits_b) * 100)


    def rs_bpp(self, hidden_image, extracted_image, D: float = 2.0):
        a = self._to_array(hidden_image).astype(uint8)
        b = self._to_array(extracted_image).astype(uint8)
        if a.shape != b.shape:
            raise ValueError(f"Tailles différentes : {a.shape} vs {b.shape}. " "Redimensionne l'image cachée à la taille de l'extraite avant de comparer.")
        acc_pct = self.bit_accuracy(hidden_image, extracted_image)
        p = 1.0 - (acc_pct / 100.0)
        return float(max(0.0, (1.0 - 2.0 * p) * D))