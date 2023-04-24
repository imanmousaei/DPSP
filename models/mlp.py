from keras.layers import Dense, Dropout, Input, Activation, BatchNormalization
from keras.models import Model

import torch.nn as nn
import torch.nn.functional as F


def DNN():
    train_input = Input(shape=(new_feature.shape[1],), name='Inputlayer')
    train_in = Dense(512, activation='relu')(train_input)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(256, activation='sigmoid')(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(128, activation='sigmoid')(train_in)
    train_in = BatchNormalization()(train_in)
    train_in = Dropout(droprate)(train_in)

    train_in = Dense(event_num)(train_in)
    out = Activation('softmax')(train_in)
    model = Model(train_input, out)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


class MLP(nn.Module):

    def __init__(self, input_size, output_size, droprate):
        super(MLP, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.droprate = droprate

        self.seq = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
            nn.Dropout(p=droprate, inplace=True),

            nn.Linear(512, 256),
            nn.Sigmoid(),
            nn.BatchNorm2d(256),
            nn.Dropout(p=droprate, inplace=True),

            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.BatchNorm2d(128),
            nn.Dropout(p=droprate, inplace=True),


            nn.Linear(128, output_size),
            nn.Softmax(output_size),
        )

    def forward(self, x):
        x = self.seq(x)

        return x
