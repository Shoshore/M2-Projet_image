from . import encoding
from . import decoding
from . import critic
from . import loss
from . import train
from . import train_dataset
from os.path import join, exists
from os import urandom
from torch import save, load, no_grad, from_numpy
from torch.cuda import is_available
from numpy import array, uint8, unpackbits, random, concatenate, zeros, packbits, frombuffer, unpackbits
from PIL import Image
from reedsolo import RSCodec, ReedSolomonError
from io import BytesIO

class Stegano_gan:

    def __init__(self, D: int = 2, not_train = True):
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
        if not_train:
            self.load_model()


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
        path = path.join(save_dir, f"steganogan_checkpoint_{self.D}.pth")
        save(checkpoint, path)
        print(f"-> Checkpoint de l'époque {epoch} sauvegardé dans : {path}")


    def load_checkpoint(self):
        device = "cpu"
        if is_available():
            device = "cuda"

        save_dir = self._get_save_dir()
        path = path.join(save_dir, f"steganogan_checkpoint_{self.D}.pth")

        if exists(path):
            checkpoint = load(path, map_location = device)
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
        path_e = join(save_dir, f"encoder_steganogan_{self.D}.pth")
        path_d = join(save_dir, f"decoder_steganogan_{self.D}.pth")
        path_c = join(save_dir, f"critic_steganogan_{self.D}.pth")

        save(self._encoder.state_dict(), path_e)
        save(self._decoder.state_dict(), path_d)
        save(self._critic.state_dict(), path_c)

        print(f"Entraînement finalisé ! Modèles finaux sauvegardés dans : {save_dir}")


    def load_model(self):
        device = self._train._device
        save_dir = self._get_save_dir()

        for name, model in [("encoder", self._encoder), ("decoder", self._decoder), ("critic", self._critic)]:
            path = join(save_dir, f"{name}_steganogan_{self.D}.pth")
            if not exists(path):
                raise FileNotFoundError(f"Poids introuvables : {path}")
            model.load_state_dict(load(path, map_location=device))
            model.to(device)

        self._encoder.eval()
        self._decoder.eval()
        self._critic.eval()
        print(f"Modèle D={self.D} chargé depuis {save_dir}")


    def _rs_layout(self, height: int, width: int, nsym: int):
        n_blocks = (self.D * height * width) // 8 // 255
        if n_blocks < 1 or nsym >= 255:
            raise ValueError("Cover trop petite ou nsym trop grand.")
        return n_blocks, 255 - nsym


    def _compress_secret(self, message_image: Image.Image, max_bytes: int):
        img = message_image.convert("RGB")
        side = min(img.size)
        while side >= 8:
            im = img.resize((side, side))
            for quality in range(90, 49, -10):
                buf = BytesIO()
                im.save(buf, format="JPEG", quality=quality)
                if buf.tell() <= max_bytes:
                    return buf.getvalue(), side, quality
            side = int(side * 0.9)
        raise ValueError("Le secret ne rentre pas, même très réduit.")


    @no_grad()
    def hide_image(self, cover_image, message_image, height : int, width : int, nsym: int = 110, filename: str = "stego.png"):
        device = self._train._device

        cover = cover_image.convert("RGB").resize((width, height))
        cover_t = from_numpy(array(cover)).permute(2, 0, 1).float() / 255.0
        cover_t = (cover_t * 2 - 1).unsqueeze(0).to(device)

        n_blocks, k = self._rs_layout(height, width, nsym)
        payload, side, quality = self._compress_secret(message_image, n_blocks * k)
        n_bytes = len(payload)
        padded = payload + urandom(n_blocks * k - n_bytes)

        rs = RSCodec(nsym)
        encoded = bytearray()
        for i in range(n_blocks):
            encoded += rs.encode(padded[i * k:(i + 1) * k])
        blocks = frombuffer(bytes(encoded), dtype=uint8).reshape(n_blocks, 255)

        stream = blocks.T.flatten()
        bits = unpackbits(stream)

        capacity = self.D * height * width
        padding = random.randint(0, 2, capacity - len(bits), dtype=uint8)
        bits = concatenate([bits, padding])

        message = from_numpy(bits).float().reshape(1, self.D, height, width).to(device)

        stego_t = self._encoder(cover_t, message).float().clamp(-1, 1)
        stego_t = (stego_t + 1) / 2
        stego_np = (stego_t[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(uint8)
        stego_img = Image.fromarray(stego_np)

        stego_img.save(join("./assets/result/ai_based_method", filename))

        infos = {"height": height, "width": width, "nsym": nsym, "n_bytes": n_bytes, "side": side, "quality": quality}
        print(f"Secret : JPEG {side}x{side} q={quality}, {n_bytes} octets sur {n_blocks * k} possibles " f"({n_blocks} blocs RS, {nsym} octets de correction par bloc).")
        return stego_img, infos


    @no_grad()
    def extract_image(self, stego_image, infos):
        device = self._train._device
        height, width, nsym, n_bytes = infos["height"], infos["width"], infos["nsym"], infos["n_bytes"]
        n_blocks, k = self._rs_layout(height=height, width=width, nsym=nsym)

        stego = stego_image.convert("RGB") 
        stego_t = from_numpy(array(stego)).permute(2, 0, 1).float() / 255.0
        stego_t = (stego_t * 2 - 1).unsqueeze(0).to(device)

        logits = self._decoder(stego_t)
        bits = (logits > 0).flatten().cpu().numpy().astype(uint8)

        stream = packbits(bits[:n_blocks * 255 * 8])
        blocks = stream.reshape(255, n_blocks).T

        rs = RSCodec(nsym)
        data, failed = bytearray(), 0
        for block in blocks:
            try:
                data += rs.decode(bytes(block))[0]
            except ReedSolomonError:
                failed += 1
                data += bytes(block[:k])
        print(f"Blocs RS non corrigeables : {failed}/{n_blocks}")

        return Image.open(BytesIO(bytes(data[:n_bytes]))).convert("RGB")


    def hide_adaptive(self, cover_image, message_image, height : int = 256, width: int = 256, nsyms=(64, 110, 140, 180), tries: int = 2, filename: str = "stego.png"):
        for nsym in nsyms:
            n_blocks, k = self._rs_layout(height=height, width=width, nsym=nsym)
            payload, _, _ = self._compress_secret(message_image, n_blocks * k)
            ref = array(Image.open(BytesIO(payload)).convert("RGB"))
            for _ in range(tries):
                stego, infos = self.hide_image(cover_image, message_image, height=height, width=width, nsym=nsym, filename=filename)
                saved = Image.open(join("./assets/result/ai_based_method", filename))
                try:
                    rec = array(self.extract_image(saved, infos))
                    if rec.shape == ref.shape and (rec == ref).all():
                        return stego, infos
                except Exception:
                    pass
        raise RuntimeError("Aucun nsym ne permet une récupération exacte pour cette cover.")


    def set_d(self, D : int):
        if D == self.D:
            return
        self.D = D
        self._encoder = encoding.Encoder(D = self.D)
        self._decoder = decoding.Decoding(D = self.D)
        self._critic = critic.Critic()
        self.load_model()
