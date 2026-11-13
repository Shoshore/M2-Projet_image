from torch.nn import Module, Sequential, Conv2d, LeakyReLU, BatchNorm2d


class Critic(Module):

    def __init__(self):
        super().__init__()

        self.first_layer = Sequential(
            Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            LeakyReLU(negative_slope = 0.01, inplace = True),
            BatchNorm2d(32)
        )

        self.second_layer = Sequential(
            Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            LeakyReLU(negative_slope = 0.01, inplace = True),
            BatchNorm2d(32)
        )

        self.third_layer = Sequential(
            Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            LeakyReLU(negative_slope = 0.01, inplace = True),
            BatchNorm2d(32)
        )


        self.fourth_layer = Conv2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, S):
        a = self.first_layer(S)
        b = self.second_layer(a)
        c = self.third_layer(b)
        d = self.fourth_layer(c)

        score = d.mean(dim=(1, 2, 3))
        return score