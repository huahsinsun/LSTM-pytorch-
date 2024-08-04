import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


# 时间序列数据集类定义
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


# 位置编码模块定义
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


# Transformer模型定义
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_layers, d_model, nhead, dim_feedforward, output_size):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer( batch_first=True,d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_size)
        self.d_model = d_model


    def forward(self, src):
        src = src * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output


# 数据加载和准备
train_df = pd.read_csv(r'C:\Users\huahs\PycharmProjects\LSTM-pytorch\dataset\train_dataset.csv')
val_df = pd.read_csv(r'C:\Users\huahs\PycharmProjects\LSTM-pytorch\dataset\val_dataset.csv')
test_df = pd.read_csv(r'C:\Users\huahs\PycharmProjects\LSTM-pytorch\dataset\test_dataset.csv')
train_data = torch.tensor(train_df['Elia Grid Load [MW]'].values, dtype=torch.float32).unsqueeze(1)
val_data = torch.tensor(val_df['Elia Grid Load [MW]'].values, dtype=torch.float32).unsqueeze(1)
test_data = torch.tensor(test_df['Elia Grid Load [MW]'].values, dtype=torch.float32).unsqueeze(1)

# 模型参数
input_size = 1
num_layers = 2
d_model = 32
nhead = 2
dim_feedforward = 256
output_size = 1
num_epochs = 20
learning_rate = 0.001
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 训练和评估函数
def train_and_evaluate(seq_length):
    # 创建数据集和数据加载器
    train_dataset = TimeSeriesDataset(train_data, seq_length)
    val_dataset = TimeSeriesDataset(val_data, seq_length)
    test_dataset = TimeSeriesDataset(test_data, seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型、损失函数和优化器
    model = TransformerModel(input_size, num_layers, d_model, nhead, dim_feedforward, output_size)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 模型训练
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()

        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        val_loss /= len(val_dataloader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_seq{seq_length}.pt')

    # 加载最优模型并进行测试
    model.load_state_dict(torch.load(f'best_model_seq{seq_length}.pt'))
    model.eval()
    predictions, true_values = [], []
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.detach().cpu().numpy())
            true_values.extend(targets.detach().cpu().numpy())

    # 计算性能指标
    mse = np.mean((np.array(predictions) - np.array(true_values)) ** 2)
    rmse = np.sqrt(mse)
    return predictions, true_values, mse, rmse


# 序列长度测试
# seq_lengths = [12, 18, 24, 30]
seq_lengths = [18,24,30,36,42]
results = [train_and_evaluate(sl) for sl in seq_lengths]

import matplotlib
# 设置matplotlib的字体为SimHei
matplotlib.rcParams['font.sans-serif'] = ['SimSun']  # 或者使用其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 结果可视化
plt.figure(figsize=(15,10))
for i, (predictions, true_values, mse, rmse) in enumerate(results):
    plt.subplot(3,2,i + 1)
    plt.plot(true_values[:96], label='真实值')
    plt.plot(predictions[:96], label='预测值')
    plt.xlim(0,85)
    plt.title(f'序列长度 {seq_lengths[i]} ')
    plt.xlabel('时间')
    plt.ylabel('归一化负荷 [MW]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{seq_lengths[i]}',dpi=350)



# 性能指标输出
for i, (predictions, true_values, mse, rmse) in enumerate(results):
    print(f'序列长度 {seq_lengths[i]} - MSE: {mse:.4f}, RMSE: {rmse:.4f}')