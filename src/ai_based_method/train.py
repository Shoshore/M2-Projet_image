from torch import cuda, bernoulli, full, no_grad
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.optim import Adam

class Train:

    def __init__(self, loss, encoder, decoder, critic):
        self._epochs = 32
        self._lr = 1e-4

        self._device = "cpu"
        if cuda.is_available():
            self._device = "cuda"
            cudnn.benchmark = True

        self._loss = loss.to(self._device)
        self._encoder = encoder.to(self._device)
        self._decoder = decoder.to(self._device)
        self._critic = critic.to(self._device)

        self.optimizer_C = Adam(self._critic.parameters(), lr = self._lr)
        self.optimizer_ED = Adam(list(self._encoder.parameters()) + list(self._decoder.parameters()), lr = self._lr)


    def generate_secret_message(self, batch_size: int, D: int, H: int, W: int):
        message = bernoulli(full((batch_size, D, H, W), 0.5))
        return message.to(self._device)


    def fit(self, train_loader: DataLoader, D: int, save_callback = None, start_epoch: int = 1):
        train_load_size = len(train_loader)
        print("Début de l'entrainnement")
        for epoch in range(start_epoch, self._epochs + 1):
            self._encoder.train()
            self._decoder.train()
            self._critic.train()

            running_loss_c = 0.0
            running_loss_ed = 0.0
            running_acc = 0.0
            # -------------
            # critic
            # -------------
            for _, cover_image in enumerate(train_loader):
                cover_image = cover_image.to(self._device)
                batch_size, _, H, W = cover_image.shape

                secret_message = self.generate_secret_message(batch_size, D, H, W)
                self.optimizer_C.zero_grad()

                with no_grad():
                    stego_image = self._encoder(cover_image, secret_message)

                critical_real = self._critic(cover_image)
                critical_fake = self._critic(stego_image)

                loss_c = self._loss.critic_loss(critical_real.float(), critical_fake.float())
                loss_c.backward()
                self.optimizer_C.step()

                for p in self._critic.parameters():
                    p.data.clamp_(-0.1, 0.1)
                # -------------
                # auto encoder
                # -------------
                self.optimizer_ED.zero_grad()

                stego_image = self._encoder(cover_image, secret_message)
                decoded_message = self._decoder(stego_image)
                critical_fake_ed = self._critic(stego_image)

                total_loss = self._loss.auto_encoding_loss(
                    decoded_message.float(),
                    secret_message,
                    cover_image,
                    stego_image.float(),
                    critical_fake_ed.float()
                )

                total_loss.backward()

                self.optimizer_ED.step()

                running_loss_c += loss_c.item()
                running_loss_ed += total_loss.item()

                running_acc += ((decoded_message.float() > 0).float() == secret_message).float().mean().item()

            avg_c = running_loss_c / train_load_size
            avg_ed = running_loss_ed / train_load_size
            avg_acc = running_acc / train_load_size
            print(f"Epoch [{epoch}/{self._epochs}] \n Critic loss: {avg_c:.4f} \n ED loss: {avg_ed:.4f} \n Accuracy: {avg_acc:.4f}")

            if save_callback is not None:
                save_callback(epoch)