import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, input_size, output_size, droprate):
        super(MLP, self).__init__()

        self.input_size = input_size
        print('input_size', input_size)
        self.output_size = output_size
        self.droprate = droprate

        self.seq = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(p=droprate),

            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.BatchNorm1d(256),
            nn.Dropout(p=droprate),

            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=droprate),

            nn.Linear(128, output_size),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        print('forward')
        x = self.seq(x)

        return x
