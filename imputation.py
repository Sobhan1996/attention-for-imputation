import math
import time

import torch
import torch.optim as optim
from transformer.Models import Encoder
from transformer.Optim import ScheduledOptim

from sklearn import preprocessing

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# import torch_xla
# import torch_xla.core.xla_model as xm


def normalize_df(df):
    x = df.values
    min_max_scalar = preprocessing.MinMaxScaler()
    x_scaled = min_max_scalar.fit_transform(x)
    return pd.DataFrame(x_scaled)


class Dataset:
    def __init__(self, source_dataset, batch_size, epochs, window_size, device, plot_file, train_data,
                 test_data, valid_data, target_column, target_min, target_max, d_inner, n_layers, n_head_, d_k, d_v,
                 n_warmup_steps, criterion, target_name, d_model, model_file=None, load_data=False, load_model=False):
        self.data_frame = self.read_dataset(source_dataset)
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device
        self.target_column = target_column
        self.window = window_size
        self.plot_file = plot_file
        self.n_layers = n_layers
        self.n_head = n_head_
        self.d_inner = d_inner
        self.warmup_step = n_warmup_steps
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.target_name = target_name
        self.input_mask = torch.ones([self.batch_size, 1, self.window], dtype=torch.int, device=device)
        self.target_max = target_max
        self.target_min = target_min
        self.model_file = model_file
        self.prev_epoch = 0
        if load_data:
            self.train_df = pd.read_csv(train_data)
            self.test_df = pd.read_csv(test_data)
            self.valid_df = pd.read_csv(valid_data)
        else:
            self.train_df, self.valid_df, self.test_df = self.organize_dataset(train_data, test_data, valid_data)

        pad_col = ['col' + str(i) for i in range(self.train_df.shape[1], self.d_model)]
        for col in pad_col:
            self.train_df[col] = 0
            self.test_df[col] = 0
            self.valid_df[col] = 0
        self.columns = self.train_df.shape[1]
        self.model = Encoder(
            n_position=200,
            d_word_vec=self.columns, d_model=self.columns, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head_, d_k=d_k, d_v=d_v,
            dropout=0).to(device)

        if load_model:
            self.model = torch.load(self.model_file)['model']
            self.model.eval()
            self.model = self.model.to(device)
            self.prev_epoch = torch.load(self.model_file)['epoch']

        self.criterion = criterion
        self.optimizer = ScheduledOptim(
            optim.Adam(self.model.parameters(), betas=(0.9, 0.98), eps=1e-09),
            2.0, self.columns, n_warmup_steps, n_step=self.prev_epoch*(math.floor(self.train_df.shape[0]/self.window * self.batch_size)))
        self.loss_list = []
        self.lr_list = []

    def read_dataset(self, source_dataset):
        return pd.read_csv(source_dataset)

    def organize_dataset(self, train_data, test_data, valid_data):
        train_df = self.data_frame
        valid_df = self.data_frame
        test_df = self.data_frame
        return train_df, valid_df, test_df

    def train(self):
        train_tensor = torch.tensor(self.train_df.values, dtype=torch.float, device=self.device)
        train_rows = self.train_df.shape[0]
        section_size = self.window * self.batch_size
        avg_loss = 0
        for i in range(self.epochs):
            chosen_idx = np.random.choice(train_rows, replace=True, size=math.floor(train_rows/10))
            imputing_df = self.train_df.copy()
            imputing_df.iloc[[j in chosen_idx for j in range(train_rows)], self.target_column] = 0
            imputing_tensor = torch.tensor(imputing_df.values, dtype=torch.float, device=self.device)

            avg_loss = 0
            lr = 0

            for j in range(math.floor(train_rows/section_size)):
                batch_imputing_tensor = imputing_tensor[j * section_size: (j+1) * section_size, :]
                batch_train_tensor = train_tensor[j * section_size: (j+1) * section_size, :]

                input_tensor = self.unsqueeze(batch_imputing_tensor)

                self.optimizer.zero_grad()

                imputed_tensor = self.squeeze(self.model(input_tensor, self.input_mask)[0])

                imputing_idx = [k in chosen_idx for k in range(j * section_size, (j+1) * section_size)]
                imputing_idx_tensor = torch.tensor(imputing_idx)

                imputed_label_tensor = imputed_tensor[imputing_idx_tensor, self.target_column]
                true_label_tensor = batch_train_tensor[imputing_idx_tensor, self.target_column]

                loss = torch.sqrt(self.criterion(imputed_label_tensor, true_label_tensor))
                # loss = self.criterion(imputed_label_tensor, true_label_tensor)

                if imputed_label_tensor.shape[0] > 0:

                    loss.backward()     #here compute engine
                    lr = self.optimizer.step_and_update_lr()

                    avg_loss = (j*avg_loss + loss) / (j+1)

            self.loss_list.append(avg_loss*(self.target_max - self.target_min))
            self.lr_list.append(10000 * lr)

            self.save_model(i)

            print(avg_loss*(self.target_max - self.target_min))

        self.draw_plots(avg_loss*(self.target_max - self.target_min))

    def validate(self):
        valid_tensor = torch.tensor(self.valid_df.values, dtype=torch.float, device=self.device)
        valid_rows = self.valid_df.shape[0]
        section_size = self.window * self.batch_size

        chosen_idx = np.random.choice(valid_rows, replace=True, size=math.floor(valid_rows/10))
        imputing_df = self.valid_df.copy()
        imputing_df.iloc[[j in chosen_idx for j in range(valid_rows)], self.target_column] = 0
        imputing_tensor = torch.tensor(imputing_df.values, dtype=torch.float, device=self.device)
        avg_loss = 0

        imputed_list = []

        for j in range(math.floor(valid_rows/section_size)):
            batch_imputing_tensor = imputing_tensor[j * section_size: (j+1) * section_size, :]
            batch_valid_tensor = valid_tensor[j * section_size: (j+1) * section_size, :]

            input_tensor = self.unsqueeze(batch_imputing_tensor)

            imputed_tensor = self.squeeze(self.model(input_tensor, self.input_mask)[0])

            imputing_idx = [k in chosen_idx for k in range(j * section_size, (j+1) * section_size)]
            imputing_idx_tensor = torch.tensor(imputing_idx)

            imputed_label_tensor = imputed_tensor[imputing_idx_tensor, self.target_column]
            true_label_tensor = batch_valid_tensor[imputing_idx_tensor, self.target_column]

            imputed_list = imputed_list + imputed_tensor[:, self.target_column].tolist()

            # loss = torch.sqrt(self.criterion(imputed_label_tensor, true_label_tensor))
            loss = self.criterion(imputed_label_tensor, true_label_tensor)

            if imputed_label_tensor.shape[0] > 0:
                avg_loss = (j*avg_loss + loss) / (j+1)

        print(avg_loss*(self.target_max - self.target_min))

        valid_list = valid_tensor[:, self.target_column].tolist()
        imputed_list = [(imputed_list[i] * (i in chosen_idx) + valid_list[i] * (i not in chosen_idx)) for i in range(len(imputed_list))]

        plt.plot(imputed_list, 'r', label="Imputed")
        plt.plot(valid_list, 'b', label="True")
        plt.legend(loc="upper right")
        plt.show()

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

    def draw_plots(self, avg_loss):
        plt.plot(self.loss_list, 'r', label="Loss")
        plt.plot(self.lr_list, 'b', label="10000 * Learning Rate")
        title = 'n_layers: ' + str(self.n_layers) + '\n' + 'n_heads: ' + str(self.n_head) + '\n' + 'd_inner: ' + str(self.d_inner) + '\n' + 'warmup_step: ' + str(self.warmup_step) + '\n' + 'd_v: ' + str(self.d_v) + '\n' + 'd_k: ' + str(self.d_k) + '\n' + 'd_model: ' + str(self.d_model) + '\n' + 'window: ' + str(self.window) + '\n' + 'target_column: ' + self.target_name + '\n' + 'Loss_function: ' + str(self.criterion) + '\n' + 'avg_loss: ' + str(float(avg_loss.data))
        plt.legend(loc="upper right", title=title)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        plt.savefig(self.plot_file + timestr, quality=90)

    def save_model(self, epoch):
        checkpoint = {'epoch': epoch, 'lr_list': self.lr_list, 'loss_list': self.loss_list,
                      'model': self.model}
        if self.model_file:
            torch.save(checkpoint, self.model_file)


class AirQualityDataset(Dataset):
    def __init__(self, source_dataset, batch_size, epochs, window_size, device, plot_file, train_data, test_data,
                 valid_data, target_column, target_min, target_max, d_inner, n_layers, n_head_, d_k, d_v,
                 n_warmup_steps, criterion, target_name, d_model, model_file, load_data, load_model):
        Dataset.__init__(self, source_dataset, batch_size, epochs, window_size, device, plot_file, train_data,
                         test_data, valid_data, target_column, target_min, target_max, d_inner, n_layers, n_head_, d_k,
                         d_v, n_warmup_steps, criterion, target_name, d_model, model_file, load_data, load_model)

    def organize_dataset(self, train_data, test_data, valid_data):
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

        self.target_max = clean_df[self.target_name].max()
        self.target_min = clean_df[self.target_name].min()

        clean_df[self.target_name] = (clean_df[self.target_name] - self.target_min) / (self.target_max - self.target_min)

        valid_df = clean_df.loc[clean_df['month'].isin([4, 8, 12])]
        train_df = clean_df.loc[~clean_df['month'].isin([4, 8, 12])]

        test_df = test_df.drop('month', axis=1)
        train_df = train_df.drop('month', axis=1)
        valid_df = valid_df.drop('month', axis=1)

        train_df = normalize_df(train_df)
        test_df = normalize_df(test_df)
        valid_df = normalize_df(valid_df)

        train_df.to_csv(train_data, index=False)
        test_df.to_csv(test_data, index=False)
        valid_df.to_csv(valid_data, index=False)

        return train_df, valid_df, test_df


class HumanActivityDataset(Dataset):
    def __init__(self, source_dataset, batch_size, epochs, window_size, device, plot_file, train_data, test_data,
                 valid_data, target_column, target_min, target_max, d_inner, n_layers, n_head_, d_k, d_v,
                 n_warmup_steps, criterion, target_name, d_model, model_file, load_data, load_model):
        Dataset.__init__(self, source_dataset, batch_size, epochs, window_size, device, plot_file, train_data,
                         test_data, valid_data, target_column, target_min, target_max, d_inner, n_layers, n_head_, d_k,
                         d_v, n_warmup_steps, criterion, target_name, d_model, model_file, load_data, load_model)

    def organize_dataset(self, train_data, test_data, valid_data):
        self.data_frame.drop('date', axis=1, inplace=True)
        self.data_frame.drop('timestamp', axis=1, inplace=True)
        self.data_frame = pd.get_dummies(self.data_frame)

        self.target_max = self.data_frame[self.target_name].max()
        self.target_min = self.data_frame[self.target_name].min()

        self.data_frame[self.target_name] = (self.data_frame[self.target_name] - self.target_min) / (self.target_max - self.target_min)

        valid_df = self.data_frame.loc[self.data_frame['Sequence_name_A01'].isin([1])]
        train_df = self.data_frame.loc[self.data_frame['Sequence_name_A01'].isin([0])]

        train_df = normalize_df(train_df)
        valid_df = normalize_df(valid_df)

        train_df.to_csv(train_data, index=False)
        valid_df.to_csv(valid_data, index=False)

        return train_df, valid_df, valid_df


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dataset = AirQualityDataset(source_dataset='./datasets/PRSA_data_2010.1.1-2014.12.31.csv', batch_size=1, epochs=10,
                            window_size=30, device=device, plot_file='./AirQualityData/AirQuality_plot',
                            model_file='./AirQualityData/model.chkpt', train_data=r'./AirQualityData/train.csv',
                            test_data=r'./AirQualityData/test.csv', valid_data=r'./AirQualityData/valid.csv',
                            load_data=False, load_model=True, target_column=0, target_min=0, target_max=994, d_inner=128,
                            n_layers=1, n_head_=1, d_k=32, d_v=32, criterion=torch.nn.MSELoss(), n_warmup_steps=5000,
                            target_name='pm2.5', d_model=32)

# dataset = HumanActivityDataset(source_dataset='./datasets/ConfLongDemo_JSI.txt', batch_size=25, epochs=100,
#                                window_size=30, device=device, plot_file='./HumanActivityData/HumanActivity_plot',
#                                model_file='./HumanActivityData/model.chkpt', train_data=r'./HumanActivityData/train.csv',
#                                test_data=r'./HumanActivityData/test.csv', valid_data=r'./HumanActivityData/valid.csv',
#                                load_data=False, load_model=False, target_column=0, target_min=-0.27869826555252075, target_max=5.75817346572876, d_inner=128,
#                                n_layers=2, n_head_=2, d_k=32, d_v=32, criterion=torch.nn.MSELoss(), n_warmup_steps=5000,
#                                target_name='x_coordinate', d_model=43)


# dataset.train()

print('---------------')
dataset.load_model = True
dataset.criterion = torch.nn.L1Loss()
dataset.validate()

# device = xm.xla_device()  #here tpu
# dataset = AirQualityDataset('./datasets/PRSA_data_2010.1.1-2014.12.31.csv', 25, 10000, 30, device)

