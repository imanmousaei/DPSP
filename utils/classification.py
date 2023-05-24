import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch
import os
import numpy as np

from utils.metrics import EarlyStopping


class Classification:
    def __init__(self, model, learning_rate, use_gpu, early_stopping=True):

        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        patience = 20

        # init model
        self.model = model
        self.device = self._acquire_device()
        self.model = self.model.to(self.device)

        # define optimizer and criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        self.checkpoint_path = 'models/checkpoints'
        if early_stopping:
            self.early_stopping = EarlyStopping(patience=patience, verbose=True)


    def _acquire_device(self):
        gpu_device_ids = '0'

        if torch.cuda.is_available() and self.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device_ids
            device = torch.device(f'cuda:{gpu_device_ids}')
            print(f'Use GPU: cuda:{gpu_device_ids}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
            
        return device

    def train(self, epochs, batch_size, train_data, validation_data):
        print('start training')
        
        train_dataset = TensorDataset(train_data[0], train_data[1])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        self.model.train()
        for epoch in range(epochs):
            train_losses = []
            for i, (batch_X, batch_y) in tqdm(enumerate(train_loader)):
                self.optimizer.zero_grad()

                probabilities = self.model(batch_X)
                # get class number of most probability
                # predictions = torch.argmax(probabilities, dim=1)
                # print('labels \n', labels, '\n', probabilities)

                loss = self.criterion(probabilities, batch_y)
                train_losses.append(loss.item())

                loss.backward()
                # nn.utils.clip_grad_norm_(
                #     model.parameters(), max_norm=4.0) # todo: tune max_norm
                self.optimizer.step()

            # todo: validation here
            validation_loss = self.validate(validation_data, batch_size)
            print(f'validation loss: {validation_loss}, train_loss: {loss}')
            self.early_stopping(self.model, validation_loss, self.checkpoint_path, epoch)
            if self.early_stopping.trigger_early_stop:
                print("Early stopping")
                break

        # best_model_path = os.path.join(checkpoint_path, 'checkpoint.pth')
        # model.load_state_dict(torch.load(best_model_path))

        plt.plot(train_losses)
        plt.savefig('train_losses.png')

    def validate(self, validation_data, batch_size):
        print('starting validation')

        val_dataset = TensorDataset(validation_data[0], validation_data[1])
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
        )
        

        validation_losses = []
        self.model.eval()

        with torch.no_grad():
            for i, (batch_X, batch_y) in enumerate(val_loader):
                probabilities = self.model(batch_X)
                
                # get class number of most probability
                # predictions = torch.argmax(probabilities, dim=1)
                # print('labels \n', labels, '\n', probabilities)

                loss = self.criterion(probabilities, batch_y)
                validation_losses.append(loss)

        total_loss = np.average(validation_losses)

        self.model.train()
        return total_loss
