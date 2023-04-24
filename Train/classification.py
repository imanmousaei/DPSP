import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torch
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

from models.mlp import MLP
from train.utils import EarlyStopping


class Classification:
    def __init__(self, model, learning_rate, use_gpu, early_stopping=True):

        self.learning_rate = learning_rate
        self.use_gpu = use_gpu
        patience = 20

        # init model
        self.model = model
        self.device = self._acquire_device()
        self.model = self.model.to(self.device)

        # get optimizer and criterion
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
        
        train_dataset = Dataset()
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        )

        self.model.train()
        for epoch in range(epochs):
            train_losses = []
            for i, item in tqdm(enumerate(train_loader)):
                self.optimizer.zero_grad()

                # move tensors to device
                for key, value in item.items():
                    item[key] = value.to(self.device)

                labels = item['labels'].flatten()
                probabilities = self.model(item)
                # get class number of most probability
                # predictions = torch.argmax(probabilities, dim=1)
                # print('labels \n', labels, '\n', probabilities)

                loss = self.criterion(probabilities, labels)
                train_losses.append(loss.item())

                loss.backward()
                # nn.utils.clip_grad_norm_(
                #     model.parameters(), max_norm=4.0) # todo: tune max_norm
                self.optimizer.step()

            # todo: validation here
            validation_loss = self.validate()
            print(f'validation loss: {validation_loss}, train_loss: {loss}')
            self.early_stopping(self.model, validation_loss, self.checkpoint_path, epoch)

        # best_model_path = os.path.join(checkpoint_path, 'checkpoint.pth')
        # model.load_state_dict(torch.load(best_model_path))

        plt.plot(train_losses)
        plt.savefig('train_losses.png')

    def validate(self):
        print('starting validation')

        

        validation_losses = []
        self.model.eval()

        with torch.no_grad():
            for i, item in enumerate(data_loader):

                for key, value in item.items():
                    item[key] = value.to(self.device)

                labels = item['labels'].flatten()
                probabilities = self.model(item)
                # get class number of most probability
                # predictions = torch.argmax(probabilities, dim=1)
                # print('labels \n', labels, '\n', probabilities)

                loss = self.criterion(probabilities, labels)
                validation_losses.append(loss)

        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss
