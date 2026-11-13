from torch.nn import Module, Sequential, Conv2d, LeakyReLU, BatchNorm2d
from torch import cat

class Decoding(Module):

    def __init__(self, D: int = 4):
        super().__init__()
        self._D = D

        self.first_block = Sequential(
            Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            LeakyReLU(negative_slope = 0.01, inplace = True),
            BatchNorm2d(32)
        )

        self.second_block = Sequential(
            Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            LeakyReLU(negative_slope = 0.01, inplace = True),
            BatchNorm2d(32)
        )

        self.third_block = Sequential(
            Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            LeakyReLU(negative_slope = 0.01, inplace = True),
            BatchNorm2d(32)
        )

        self.fourth_block = Sequential(
            Conv2d(in_channels = 96, out_channels = self._D, kernel_size = 3, stride = 1, padding = 1),
        )


    def _cat(self, tensor_list):
        return cat(tensor_list, dim = 1)


    def forward(self, cipher_image):
        a = self.first_block(cipher_image)
        b = self.second_block(a)
        c = self.third_block(self._cat([a, b]))
        M = self.fourth_block(self._cat([a, b, c]))
        return M