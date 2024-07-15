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


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 读取测试集的数据
test_df = pd.read_csv('dataset/test_dataset.csv')

# 将数据集转换为PyTorch的Tensor
test_data = torch.tensor(test_df['Elia Grid Load [MW]'].values, dtype=torch.float32).unsqueeze(1)

# 创建测试集的数据集对象
seq_length = 24
test_dataset = TimeSeriesDataset(test_data, seq_length)

# 创建数据加载器
batch_size = 32
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义模型参数
input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1

# 创建模型实例
model = LSTMModel(input_size, hidden_size, num_layers, output_size)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载已训练好的模型参数
model.load_state_dict(torch.load('best_model.pt'))
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