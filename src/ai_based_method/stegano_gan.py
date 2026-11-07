lazy import encoding
lazy import decoding
lazy import critic


class stegano_gan:

    def __init__(self, width : int, height : int, D: int):
        self._width = width
        self._height = height
        self._encoder = encoding.Encoder(D)
        self._decoder = decoding.Decoding(D)
        self._critic = critic.Critic()
