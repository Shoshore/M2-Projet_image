from torch import nn

class Critic(nn.Module):

    def __init__(self):
        super().__init__()

        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.01, inplace = True)
        )

        self.second_layer = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.01, inplace = True)
        )

        self.third_layer = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope = 0.01, inplace = True)
        )


        self.fourth_layer = nn.Conv2d(in_channels = 32, out_channels = 1, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, S):
        a = self.first_layer(S)
        b = self.second_layer(a)
        c = self.third_layer(b)
        d = self.fourth_layer(c)

        score = d.mean(dim=(1, 2, 3))
        return score