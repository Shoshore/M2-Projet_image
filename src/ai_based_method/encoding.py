from torch import nn
import torch

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self._D = 0

        self.first_block = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.01, inplace = True)
        )

        self.second_block = nn.Sequential(
            nn.Conv2d(in_channels = 32 + self._D, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.01, inplace = True)
        )

        self.third_block = nn.Sequential(
            nn.Conv2d(in_channels = 64 + self._D, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.01, inplace = True)
        )


        self.fourth_block = nn.Sequential(
            nn.Conv2d(in_channels = 96 + self._D, out_channels = 3, kernel_size = 3, stride = 1, padding = 1),
        )


    def cat(self, tensor_list):
        return torch.cat(tensor_list, dim=1)



    def set_D(self, D : int):
        self._D = D


    def forward(self, cover_image, secret_image):
        a = self.first_block(cover_image)
        b = self.second_block(self.cat([a, secret_image]))
        c = self.third_block(self.cat([a, b, secret_image]))
        d = self.fourth_block(self.cat([a, b, c, secret_image]))
        S = cover_image
        return S