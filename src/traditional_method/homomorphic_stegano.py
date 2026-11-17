import math
import random
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
from PIL import Image

class AES_ECB:
    def __init__(self, key: bytes):
        assert len(key) == 32, "AES key must be 256 bits (32 bytes)"
        self.key = key

    def encrypt_array(self, raw_bytes: bytes) -> np.ndarray:
        cipher = AES.new(self.key, AES.MODE_ECB)
        size_of_full_blocks = len(raw_bytes) // AES.block_size * AES.block_size
        if size_of_full_blocks == 0:
            raise ValueError("the secret image must contain at least 16 bytes")

        encrypted_bytes = cipher.encrypt(raw_bytes[:size_of_full_blocks])

        tail = raw_bytes[size_of_full_blocks:]
        if tail:
            mask = cipher.encrypt(encrypted_bytes[-AES.block_size:])[:len(tail)]
            encrypted_bytes += bytes(a ^ b for a, b in zip(tail, mask))

        return np.frombuffer(encrypted_bytes, dtype=np.uint8)

    def decrypt_array(self, encrypted_bytes: np.ndarray) -> bytes:
        cipher = AES.new(self.key, AES.MODE_ECB)
        encrypted_bytes = encrypted_bytes.tobytes()
        size_of_full_blocks = len(encrypted_bytes) // AES.block_size * AES.block_size

        raw_bytes = cipher.decrypt(encrypted_bytes[:size_of_full_blocks])

        tail = encrypted_bytes[size_of_full_blocks:]
        if tail:
            mask = cipher.encrypt(encrypted_bytes[size_of_full_blocks - AES.block_size:size_of_full_blocks])[:len(tail)]
            raw_bytes += bytes(a ^ b for a, b in zip(tail, mask))

        return raw_bytes

class PaillierCryptosystem:
    def __init__(self, p: int, q: int):
        self.p = p
        self.q = q
        self.n = p * q
        self.n_squared = self.n * self.n
        self.g = self.n + 1
        self.lam = math.lcm(p - 1, q - 1)
        self.mu = pow(self.__L(pow(self.g, self.lam, self.n_squared)), -1, self.n)

        self.alpha_prime = self.n.bit_length() - 1
        self.beta_prime = 2 * self.alpha_prime
        if self.alpha_prime % 4:
            raise ValueError("floor(log2(n)) must be a multiple of 4 "
                             "(encrypted blocks must be a whole number of pixels)")
        self.pixels_per_encrypted_block = self.beta_prime // 8

    def __L(self, x: int) -> int:
        return (x - 1) // self.n

    def encrypt_block(self, m: int, r_n: int = None) -> int:
        return (pow(self.g, m, self.n_squared) * r_n) % self.n_squared

    def decrypt_block(self, c: int) -> int:
        return (self.__L(pow(c, self.lam, self.n_squared)) * self.mu) % self.n

    def r_candidates(self, max_iterations: int = 500, seed: int = 0):
        phi = (self.p - 1) * (self.q - 1)
        if phi <= max_iterations:
            rs = [r for r in range(1, self.n) if math.gcd(r, self.n) == 1]
        else:
            rng, rs = random.Random(seed), []
            while len(rs) < max_iterations:
                r = rng.randrange(1, self.n)
                if math.gcd(r, self.n) == 1:
                    rs.append(r)
        return [pow(r, self.n, self.n_squared) for r in rs]

def block_distance(e_i: int, t_i: int, pixels_per_encrypted_block: int) -> int:
    a = np.frombuffer(e_i.to_bytes(pixels_per_encrypted_block, "big"), dtype=np.uint8).astype(np.int64)
    b = np.frombuffer(t_i.to_bytes(pixels_per_encrypted_block, "big"), dtype=np.uint8).astype(np.int64)
    return int(((a - b) ** 2).sum())

def encryption_transformation_of_plaintext_block(m_i: int, t_i: int, paillier: PaillierCryptosystem, candidates):
    min_dist = float('inf')
    best_e_i = None

    for r_n in candidates:
        e_i = paillier.encrypt_block(m_i, r_n)
        if e_i >= 1 << paillier.beta_prime:
            continue
        dist = block_distance(e_i, t_i, paillier.pixels_per_encrypted_block)
        if dist < min_dist:
            min_dist = dist
            best_e_i = e_i

    if best_e_i is None:
        raise RuntimeError("no value of r gives an encrypted block of at most beta' bits")
    return best_e_i

def load_grayscale_image(path: str):
    image = Image.open(path).convert('L')
    np_image = np.array(image, dtype=np.uint8)
    shape = np_image.shape

    return np_image, shape

def load_color_image(path: str) -> np.ndarray:
    image = Image.open(path).convert('RGB')
    np_image = np.asarray(image, dtype=np.uint8)

    return np_image

def save_image(image_array: np.ndarray, shape: tuple, path: str):
    reshaped_array = np.asarray(image_array, dtype=np.uint8).reshape(shape)
    image = Image.fromarray(reshaped_array)
    image.save(path)

def split_into_blocks(data: np.ndarray, alpha_prime: int) -> np.ndarray:
    bits = np.unpackbits(data)
    number_of_blocks = -(-len(bits) // alpha_prime)
    bits = np.pad(bits, (0, number_of_blocks * alpha_prime - len(bits))).reshape(number_of_blocks, alpha_prime)
    weights = [1 << (alpha_prime - 1 - j) for j in range(alpha_prime)]
    if alpha_prime <= 62:
        return bits.astype(np.int64) @ np.array(weights, dtype=np.int64)
    return bits.astype(object) @ np.array(weights, dtype=object)

def join_blocks(blocks: np.ndarray, alpha_prime: int, n_bytes: int) -> np.ndarray:
    shifts = np.arange(alpha_prime - 1, -1, -1).astype(blocks.dtype)
    bits = ((blocks[:, None] >> shifts[None, :]) & 1).astype(np.uint8).ravel()
    return np.packbits(bits[:8 * n_bytes])

def encrypt_transform(secret_image: np.ndarray, host_image: np.ndarray, aes: AES_ECB, paillier: PaillierCryptosystem, max_iterations: int = 500) -> np.ndarray:
    encrypted = aes.encrypt_array(np.ascontiguousarray(secret_image).ravel().tobytes())

    m_blocks = split_into_blocks(encrypted, paillier.alpha_prime)

    host_stream = np.ascontiguousarray(host_image).ravel().astype(np.uint8)
    n_pixels_used = len(m_blocks) * paillier.pixels_per_encrypted_block
    if n_pixels_used > len(host_stream):
        raise ValueError("host image too small: %d values needed, %d available" % (n_pixels_used, len(host_stream)))
    t_blocks = host_stream[:n_pixels_used].reshape(-1, paillier.pixels_per_encrypted_block)

    candidates = paillier.r_candidates(max_iterations)
    e_blocks = [encryption_transformation_of_plaintext_block(int(m_i), int.from_bytes(t_i.tobytes(), "big"), paillier, candidates) for m_i, t_i in zip(m_blocks, t_blocks)]

    transformed_pixels = host_stream.copy()
    transformed_pixels[:n_pixels_used] = np.frombuffer(b"".join(int(e_i).to_bytes(paillier.pixels_per_encrypted_block, "big") for e_i in e_blocks), dtype=np.uint8)
    return transformed_pixels.reshape(host_image.shape)

def decrypt_transform(transformed_image: np.ndarray, secret_shape: tuple, aes: AES_ECB, paillier: PaillierCryptosystem) -> np.ndarray:
    n_encrypted = int(np.prod(secret_shape))
    n_blocks = -(-8 * n_encrypted // paillier.alpha_prime)
    stream = np.ascontiguousarray(transformed_image).ravel().astype(np.uint8).tobytes()

    m_blocks = [paillier.decrypt_block(int.from_bytes(stream[i * paillier.pixels_per_encrypted_block:(i + 1) * paillier.pixels_per_encrypted_block], "big")) for i in range(n_blocks)]
    dtype = np.int64 if paillier.alpha_prime <= 62 else object
    encrypted = join_blocks(np.array(m_blocks, dtype=dtype), paillier.alpha_prime, n_encrypted)

    raw = np.frombuffer(aes.decrypt_array(encrypted), dtype=np.uint8)
    return raw.reshape(secret_shape)

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255 ** 2 / mse)

def entropy(image: np.ndarray) -> float:
    p = np.bincount(image.ravel(), minlength=256) / image.size
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())

if __name__ == "__main__":
    import sys

    aes = AES_ECB(get_random_bytes(32))
    paillier = PaillierCryptosystem(13, 23)

    secret, secret_shape = load_grayscale_image(sys.argv[1])
    host, _ = load_grayscale_image(sys.argv[2])

    print("host values needed (2 x secret): %d, available: %d" % (2 * int(np.prod(secret_shape)), host.size))

    transformed = encrypt_transform(secret, host, aes, paillier)
    recovered = decrypt_transform(transformed, secret_shape, aes, paillier)

    print("PSNR(host, transformed) = %.2f dB" % psnr(host, transformed))
    print("Entropy host = %.4f, transformed = %.4f" % (entropy(host), entropy(transformed)))
    print("Lossless reconstruction:", np.array_equal(secret, recovered))
    save_image(transformed, transformed.shape, "transformed.png")
    save_image(recovered, recovered.shape, "recovered.png")


#number_of_pixels = original_shape[0] * original_shape[1]

#save_image(encrypted_image[:number_of_pixels].astype(np.uint8), original_shape, "test_encrypted.jpeg")

#decrypted_image = np.frombuffer(aes.decrypt_array(encrypted_image), dtype=np.uint8)
#save_image(decrypted_image, original_shape, "test_decrypted.jpeg")

# color_image = load_color_image("images.jpeg")
# im = Image.fromarray(color_image)
# im.save("test_color.jpeg")