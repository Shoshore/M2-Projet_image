from torch.nn import Module, Sequential, Conv2d, LeakyReLU, BatchNorm2d
from torch import cat

class Encoder(Module):

    def __init__(self, D: int = 4):
        super().__init__()
        self._D = D

        self.first_block = Sequential(
            Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            LeakyReLU(negative_slope = 0.01, inplace = True),
            BatchNorm2d(32)
        )

        self.second_block = Sequential(
            Conv2d(in_channels = 32 + self._D, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            LeakyReLU(negative_slope = 0.01, inplace = True),
            BatchNorm2d(32)
        )

        self.third_block = Sequential(
            Conv2d(in_channels = 64 + self._D, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            LeakyReLU(negative_slope = 0.01, inplace = True),
            BatchNorm2d(32)
        )

        self.fourth_block = Sequential(
            Conv2d(in_channels = 96 + self._D, out_channels = 3, kernel_size = 3, stride = 1, padding = 1),
        )


    def _cat(self, tensor_list):
        return cat(tensor_list, dim=1)


    def forward(self, cover_image, secret_image):
        a = self.first_block(cover_image)
        b = self.second_block(self._cat([a, secret_image]))
        c = self.third_block(self._cat([a, b, secret_image]))
        d = self.fourth_block(self._cat([a, b, c, secret_image]))
        S = cover_image + d
        return S