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
# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_layers, d_model, nhead, dim_feedforward, output_size):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
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

# 将数据集转换为PyTorch的Tensor
test_data = torch.tensor(test_df['Elia Grid Load [MW]'].values, dtype=torch.float32).unsqueeze(1)

# 创建测试集的数据集对象
seq_length = 24
test_dataset = TimeSeriesDataset(test_data, seq_length)

# 创建数据加载器
batch_size = 32
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



input_size = 1
num_layers = 2
d_model = 64
nhead = 8
dim_feedforward = 256
output_size = 1


# 创建模型实例
model = TransformerModel(input_size, num_layers, d_model, nhead, dim_feedforward, output_size)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载已训练好的模型参数
model.load_state_dict(torch.load('../best_model.pt'))
model.to(device)
model.eval()

# 在测试集上进行预测
predictions = []
true_values = []

with torch.no_grad():
    for inputs, targets in test_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        predictions.append(outputs.detach().cpu().numpy())
        true_values.append(targets.detach().cpu().numpy())

# 将预测结果和真实值转换为一维数组
predictions = np.concatenate(predictions).flatten()
true_values = np.concatenate(true_values).flatten()

print(predictions.shape)
print(true_values.shape)

# 计算均方误差（MSE）和均方根误差（RMSE）
mse = np.mean((predictions - true_values) ** 2)
rmse = np.sqrt(mse)

print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

# 绘制预测值和真实值的曲线图
plt.figure(figsize=(10, 6))
plt.plot(test_df.index[seq_length:], true_values, label='True Values')
plt.plot(test_df.index[seq_length:], predictions, label='Predicted Values')
plt.xlabel('Time')
plt.ylabel('Load [MW]')
plt.title('Predicted vs True Values')
plt.legend()
plt.show()

print("predict:{}".format(predictions))
print(predictions.shape)

# 假设 predictions 的前 96 个点是未来的预测值
n_future = [x for x in range(96)]
future_predictions = predictions[:96]

# 取出 true_values 的前 96 个真实值
true_last_96 = true_values[:96]

# 绘制预测值和真实值的曲线图
plt.figure(figsize=(10, 6))
plt.plot(range(len(true_last_96)), true_last_96, label='True Values')
plt.plot(range(len(true_last_96)), future_predictions, label='Predicted Future Values', color='red')
plt.xlabel('Time')
plt.ylabel('Load [MW]')
plt.title('Predicted vs True Values with Future Predictions')
plt.legend()
plt.show()