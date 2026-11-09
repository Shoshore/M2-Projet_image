lazy from torch import nn
lazy import torch
# from torch import nn
# import torch


class Decoding(nn.Module):

    def __init__(self):
        super().__init__()

        self._D = 0

        self.first_block = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.01, inplace = True)
        )

        self.second_block = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.01, inplace = True)
        )

        self.third_block = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.01, inplace = True)
        )

        self.fourth_block = nn.Sequential(
            nn.Conv2d(in_channels = 96, out_channels = self._D, kernel_size = 3, stride = 1, padding = 1),
        )


    def cat(self, tensor_list):
        return torch.cat(tensor_list, dim = 1)


    def set_D(self, D : int):
        self._D = D


    def forward(self, cipher_image):
        a = self.first_block(cipher_image)
        b = self.second_block(a, cipher_image)
        c = self.third_block(self.cat([a, b]))
        M = self.fourth_block(self.cat([a, b, c]))
        return M
