lazy import encoding
lazy import decoding
lazy import critic
lazy import loss
lazy import train
# import encoding
# import decoding
# import critic
# import loss
class Stegano_gan:

    def __init__(self):
        self._encoder = encoding.Encoder()
        self._decoder = decoding.Decoding()
        self._critic = critic.Critic()
        self._loss = loss.Loss()
        self._train = train.Train(self._loss)


