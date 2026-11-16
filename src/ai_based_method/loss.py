from torch.nn import Module, BCEWithLogitsLoss, MSELoss

class Loss(Module):

    def __init__(self):
        super().__init__()
        self.bce_loss = BCEWithLogitsLoss()
        self.mse_loss = MSELoss()

    def loss_decoder(self, output_D, M):
       return self.bce_loss(output_D, M)

    
    def loss_stego(self, cover_image, encoded_image):
        return self.mse_loss(encoded_image, cover_image)

    
    def auto_encoding_loss(self, output_D, M, cover_image, encoded_image, critic_encoded):
        L_d = self.loss_decoder(output_D, M)
        L_s = self.loss_stego(cover_image, encoded_image)
        L_r = critic_encoded.mean()
        total_loss = L_d + 100.0 * L_s + L_r # Why 100 ? idk they did on their repo and it improve result
        return total_loss

    
    def critic_loss(self, critic_cover, critic_encoded): 
        return critic_cover.mean() - critic_encoded.mean()