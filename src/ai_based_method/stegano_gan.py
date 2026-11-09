import encoding
import decoding
import critic


class Stegano_gan:

    def __init__(self):
        self.epoch = 32
        
        self._encoder = encoding.Encoder()
        self._decoder = decoding.Decoding()
        self._critic = critic.Critic()


oui = Stegano_gan()