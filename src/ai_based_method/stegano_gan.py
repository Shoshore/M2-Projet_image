import os
lazy from . import encoding
lazy from . import decoding
lazy from . import critic
lazy from . import loss
lazy from . import train
lazy from . import train_dataset
lazy import torch

# from . import encoding
# from . import decoding
# from . import critic
# from . import loss
# from . import train
# from . import train_dataset
# import torch
class Stegano_gan:

    def __init__(self, D: int = 4):
        self.D = D
        self._encoder = encoding.Encoder(D = self.D)
        self._decoder = decoding.Decoding(D = self.D)
        self._critic = critic.Critic()
        self._loss = loss.Loss()
        self._train = train.Train(self._loss, self._encoder, self._decoder, self._critic)
        self._train_dataset = train_dataset.Train_dataset()

        self._trained_encoder = None
        self._trained_decoder = None
        self._trained_critic = None

    def _get_save_dir(self):
        if os.path.exists('/content'):
            os.makedirs('/content/weights', exist_ok=True)
            return '/content/weights'
        else:
            os.makedirs('./src/ai_based_method', exist_ok=True)
            return './src/ai_based_method'

    def trainning(self):
        self._train.fit(self._train_dataset.train_dataset(), self.D)

        save_dir = self._get_save_dir()

        path_e = os.path.join(save_dir, f"encoder_steganogan_{self.D}.pth")
        path_d = os.path.join(save_dir, f"decoder_steganogan_{self.D}.pth")
        path_c = os.path.join(save_dir, f"critic_steganogan_{self.D}.pth")

        torch.save(self._encoder.state_dict(), path_e)
        torch.save(self._decoder.state_dict(), path_d)
        torch.save(self._critic.state_dict(), path_c)

        print(f"Train finished! Models saved in: {save_dir}")


    def load_train(self):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        save_dir = self._get_save_dir()

        path_e = os.path.join(save_dir, f"encoder_steganogan_{self.D}.pth")
        path_d = os.path.join(save_dir, f"decoder_steganogan_{self.D}.pth")
        path_c = os.path.join(save_dir, f"critic_steganogan_{self.D}.pth")

        self._trained_encoder = self._encoder.load_state_dict(torch.load(path_e, map_location = device))
        self._encoder.eval()

        self._trained_decoder = self._decoder.load_state_dict(torch.load(path_d, map_location = device))
        self._decoder.eval()

        self._trained_critic = self._critic.load_state_dict(torch.load(path_c, map_location = device))
        self._critic.eval()
        print("Models loaded successfully!")