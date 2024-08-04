
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

# 自定义时间序列数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_length]
        y = self.data[index + self.seq_length]
        return x, y

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_layers, d_model, nhead, dim_feedforward, output_size):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                        dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_size)
        self.d_model = d_model

    def forward(self, src):
        src = src * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output

# 读取测试集的数据
test_df = pd.read_csv(r'C:\Users\huahs\PycharmProjects\LSTM-pytorch\dataset\test_dataset.csv')
test_data = torch.tensor(test_df['Elia Grid Load [MW]'].values, dtype=torch.float32).unsqueeze(1)
test_dataset = TimeSeriesDataset(test_data, seq_length)
batch_size = 32
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

input_size = 1
num_layers = 2
d_model = 64
nhead = 8
dim_feedforward = 256
output_size = 1

mse_list = []
rmse_list = []

for _ in range(1):
    model = TransformerModel(input_size, num_layers, d_model, nhead, dim_feedforward, output_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('../best_model.pt'))
    model.to(device)
    model.eval()

    predictions = []
    true_values = []

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            predictions.append(outputs.detach().cpu().numpy())
            true_values.append(targets.detach().cpu().numpy())

    predictions = np.concatenate(predictions).flatten()
    true_values = np.concatenate(true_values).flatten()

    mse = np.mean((predictions - true_values) ** 2)
    rmse = np.sqrt(mse)
    mse_list.append(mse)
    rmse_list.append(rmse)

mse_array = np.array(mse_list)
rmse_array = np.array(rmse_list)

mean_mse = np.mean(mse_array)
std_mse = np.std(mse_array)
mean_rmse = np.mean(rmse_array)
std_rmse = np.std(rmse_array)

print(f'MSE: Mean = {mean_mse}, Standard Deviation = {std_mse}')
print(f'RMSE: Mean = {mean_rmse}, Standard Deviation = {std_rmse}')




