from . import encoding
from . import decoding
from . import critic
from . import loss
from . import train
from . import train_dataset
import os
import torch

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
        self.current_epoch = 1

    def _get_save_dir(self):
            return './src/ai_based_method/data/weights'

    def save_checkpoint(self, epoch):
        save_dir = self._get_save_dir()
        checkpoint = {
            'epoch': epoch,
            'encoder_state': self._encoder.state_dict(),
            'decoder_state': self._decoder.state_dict(),
            'critic_state': self._critic.state_dict(),
            'opt_ED_state': self._train.optimizer_ED.state_dict(),
            'opt_C_state': self._train.optimizer_C.state_dict()
        }
        path = os.path.join(save_dir, f"steganogan_checkpoint_{self.D}.pth")
        torch.save(checkpoint, path)
        print(f"-> Checkpoint de l'époque {epoch} sauvegardé dans : {path}")

    def load_checkpoint(self):
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

        save_dir = self._get_save_dir()
        path = os.path.join(save_dir, f"steganogan_checkpoint_{self.D}.pth")

        if os.path.exists(path):
            checkpoint = torch.load(path, map_location = device)
            self._encoder.load_state_dict(checkpoint['encoder_state'])
            self._decoder.load_state_dict(checkpoint['decoder_state'])
            self._critic.load_state_dict(checkpoint['critic_state'])
            self._train.optimizer_ED.load_state_dict(checkpoint['opt_ED_state'])
            self._train.optimizer_C.load_state_dict(checkpoint['opt_C_state'])
            self.current_epoch = checkpoint['epoch'] + 1
            print(f"Checkpoint chargé avec succès ! Reprise prévue à l'époque : {self.current_epoch}")
            return True
        else:
            print("Aucun checkpoint trouvé. L'entraînement commencera à l'époque 1.")
            return False

    def trainning(self, resume=False):
        if resume:
            self.load_checkpoint()

        self._train.fit(
            self._train_dataset.train_dataset(),
            self.D,
            save_callback=self.save_checkpoint,
            start_epoch=self.current_epoch
        )

        save_dir = self._get_save_dir()
        path_e = os.path.join(save_dir, f"encoder_steganogan_{self.D}.pth")
        path_d = os.path.join(save_dir, f"decoder_steganogan_{self.D}.pth")
        path_c = os.path.join(save_dir, f"critic_steganogan_{self.D}.pth")

        torch.save(self._encoder.state_dict(), path_e)
        torch.save(self._decoder.state_dict(), path_d)
        torch.save(self._critic.state_dict(), path_c)

        print(f"Entraînement finalisé ! Modèles finaux sauvegardés dans : {save_dir}")