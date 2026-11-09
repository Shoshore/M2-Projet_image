lazy from torch import nn

# from torch import nn


class Train(nn.Module):

    def __init__(self, loss):
        self.epoch = 32
        self._loss = loss