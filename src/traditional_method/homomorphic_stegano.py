import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from PIL import Image

class AES_ECB:
    def __init__(self, key: bytes):
        assert len(key) == 32, "AES key must be 256 bits (32 bytes)"
        self.key = key

    def encrypt_array(self, raw_bytes: bytes) -> np.ndarray:
        cipher = AES.new(self.key, AES.MODE_ECB)
        padded_bytes = pad(raw_bytes, AES.block_size)
        encrypted_bytes = cipher.encrypt(padded_bytes)

        return np.frombuffer(encrypted_bytes, dtype=np.uint8)

    def decrypt_array(self, encrypted_bytes: np.ndarray) -> bytes:
        cipher = AES.new(self.key, AES.MODE_ECB)
        decrypted_bytes = cipher.decrypt(encrypted_bytes.tobytes())

        return unpad(decrypted_bytes, AES.block_size)


def load_grayscale_images(host_path: str, secret_path: str):
    host_image = Image.open(host_path).convert('L')
    secret_image = Image.open(secret_path).convert('L')

    host_np = np.array(host_image, dtype=np.uint8)
    secret_np = np.array(secret_image, dtype=np.uint8)

    return host_np, secret_np