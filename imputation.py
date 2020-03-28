import argparse
import math
import time
import dill as pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Field, Dataset, BucketIterator
from torchtext.datasets import TranslationDataset

import transformer.Constants as Constants
from transformer.Models import Transformer
from transformer.Models import Encoder
from transformer.Optim import ScheduledOptim

from sklearn import preprocessing

import pandas as pd
import numpy as np

import torch_xla
import torch_xla.core.xla_model as xm

def normalize_df(df):
    x = df.values
    min_max_scalar = preprocessing.MinMaxScaler()
    x_scaled = min_max_scalar.fit_transform(x)
    return pd.DataFrame(x_scaled)


class Dataset:
    def __init__(self, source_dataset, batch_size, epochs, window_size, device):
        self.data_frame = self.read_dataset(source_dataset)
        # self.data_frame = self.data_frame.iloc[0:80, :]
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.window = window_size
        self.input_mask = torch.ones([self.batch_size, 1, self.window], dtype=torch.int)
        self.target_max = 0
        self.target_min = 0
        self.train_df, self.valid_df, self.test_df = self.organize_dataset()
        self.columns = self.train_df.shape[1]
        self.model = Encoder(
            n_position=200,
            d_word_vec=self.columns, d_model=self.columns, d_inner=64,
            n_layers=2, n_head=2, d_k=8, d_v=8,
            dropout=0.1)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = ScheduledOptim(
            optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            2.0, self.columns, 4000)
        self.train()

    def read_dataset(self, source_dataset):
        pass

    def organize_dataset(self):
        train_df = self.data_frame
        valid_df = self.data_frame
        test_df = self.data_frame
        return train_df, valid_df, test_df

    def train(self):
        pass

    def unsqueeze(self, batch_tensor):
        temp_tensor = torch.zeros((self.batch_size, self.window, self.columns), dtype=torch.float, device=self.device)
        for i in range(self.batch_size):
            temp_tensor[i, :, :] = batch_tensor[i*self.window:(i+1)*self.window, :]
        return temp_tensor

    def squeeze(self, predict_tensor):
        temp_tensor = torch.zeros((self.batch_size * self.window, self.columns), dtype=torch.float, device=self.device)
        for i in range(self.batch_size):
            temp_tensor[i*self.window:(i+1)*self.window, :] = predict_tensor[i, :, :]
        return temp_tensor


class AirQualityDataset(Dataset):
    def __init__(self, source_dataset, batch_size, epochs, window_size, device):
        Dataset.__init__(self, source_dataset, batch_size, epochs, window_size, device)

    def read_dataset(self, source_dataset):
        return pd.read_csv(source_dataset)

    def organize_dataset(self):
        self.data_frame['sin_hour'] = np.sin(2*np.pi*self.data_frame.hour/24)
        self.data_frame['cos_hour'] = np.cos(2*np.pi*self.data_frame.hour/24)
        self.data_frame.drop('hour', axis=1, inplace=True)

        self.data_frame['sin_month'] = np.sin(2*np.pi*self.data_frame.month/24)
        self.data_frame['cos_month'] = np.cos(2*np.pi*self.data_frame.month/24)

        self.data_frame.drop('No', axis=1, inplace=True)
        self.data_frame.drop('day', axis=1, inplace=True)

        self.data_frame['year'] = self.data_frame['year'].astype('category', copy=False)

        self.data_frame = pd.get_dummies(self.data_frame)

        test_df = self.data_frame.loc[np.isnan(self.data_frame['pm2.5'])]

        clean_df = self.data_frame.loc[~np.isnan(self.data_frame['pm2.5'])]

        self.target_max = clean_df['pm2.5'].max()
        self.target_min = clean_df['pm2.5'].min()

        clean_df['pm2.5'] = (clean_df['pm2.5'] - self.target_min) / (self.target_max - self.target_min)

        valid_df = clean_df.loc[clean_df['month'].isin([4, 8, 12])]
        train_df = clean_df.loc[~clean_df['month'].isin([4, 8, 12])]

        test_df = test_df.drop('month', axis=1)
        train_df = train_df.drop('month', axis=1)
        valid_df = valid_df.drop('month', axis=1)

        test_df = normalize_df(test_df)
        train_df = normalize_df(train_df)
        valid_df = normalize_df(valid_df)

        return train_df, valid_df, test_df

    def train(self):
        train_tensor = torch.tensor(self.train_df.values, dtype=torch.float, device=self.device)
        train_rows = self.train_df.shape[0]
        section_size = self.window * self.batch_size
        for i in range(self.epochs):
            chosen_idx = np.random.choice(train_rows, replace=True, size=math.floor(train_rows/10))
            imputing_df = self.train_df.copy()
            imputing_df.loc[[j in chosen_idx for j in range(train_rows)], 0] = 0
            imputing_tensor = torch.tensor(imputing_df.values, dtype=torch.float, device=self.device)
            avg_loss = 0

            for j in range(math.floor(train_rows/section_size)):
                batch_imputing_tensor = imputing_tensor[j * section_size: (j+1) * section_size, :]
                batch_train_tensor = train_tensor[j * section_size: (j+1) * section_size, :]

                input_tensor = self.unsqueeze(batch_imputing_tensor)

                self.optimizer.zero_grad()

                imputed_tensor = self.squeeze(self.model(input_tensor, self.input_mask)[0])

                imputing_idx = [k in chosen_idx for k in range(j * section_size, (j+1) * section_size)]
                imputing_idx_tensor = torch.tensor(imputing_idx)

                imputed_label_tensor = imputed_tensor[imputing_idx_tensor, 0]
                true_label_tensor = batch_train_tensor[imputing_idx_tensor, 0]

                loss = torch.sqrt(self.criterion(imputed_label_tensor, true_label_tensor))
                loss.backward()
                self.optimizer.step_and_update_lr()

                avg_loss = (j*avg_loss + loss) / (j+1)
            print(avg_loss*(self.target_max - self.target_min))


# dataset = AirQualityDataset('./datasets/PRSA_data_2010.1.1-2014.12.31.csv', 25, 10000, 30, torch.device("cpu"))


device = xm.xla_device()
dataset = AirQualityDataset('./datasets/PRSA_data_2010.1.1-2014.12.31.csv', 25, 10000, 30, device)





# x = torch.zeros([256, 30, 10], dtype=torch.int32)
# input_mask = torch.ones([256, 1, 30], dtype=torch.int)
#
#
# model = Encoder(
#     n_position=200,
#     d_word_vec=10, d_model=10, d_inner=512,
#     n_layers=2, n_head=4, d_k=64, d_v=64,
#     dropout=0.1)
#
# y = model(x, input_mask)
#
#
# print(y)
# print(y[0].size())
#
# print('------------------')
#
# y = model(x, input_mask)
#
#
# print(y)
# print(y[0].size())