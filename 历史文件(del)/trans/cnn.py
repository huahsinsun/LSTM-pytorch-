import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt




class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        x = self.data[index:index + self.seq_length]
        y = self.data[index + self.seq_length]
        return x.transpose(0, 1), y  # 调整形状，使其符合 Conv1d 的预期

seq_length = 24
batch_size = 32

# 数据加载和准备
train_df = pd.read_csv(r'C:\Users\huahs\PycharmProjects\LSTM-pytorch\dataset\train_dataset.csv')
val_df = pd.read_csv(r'C:\Users\huahs\PycharmProjects\LSTM-pytorch\dataset\val_dataset.csv')
test_df = pd.read_csv(r'C:\Users\huahs\PycharmProjects\LSTM-pytorch\dataset\test_dataset.csv')
train_data = torch.tensor(train_df['Elia Grid Load [MW]'].values, dtype=torch.float32).unsqueeze(1)
val_data = torch.tensor(val_df['Elia Grid Load [MW]'].values, dtype=torch.float32).unsqueeze(1)
test_data = torch.tensor(test_df['Elia Grid Load [MW]'].values, dtype=torch.float32).unsqueeze(1)

train_dataset = TimeSeriesDataset(train_data, seq_length)
val_dataset = TimeSeriesDataset(val_data, seq_length)
test_dataset = TimeSeriesDataset(test_data, seq_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 定义CNN模型

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # 第一层卷积
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)

        # 第二层卷积
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)

        # 最大池化层
        self.pool = nn.MaxPool1d(2)

        # 全连接层
        # 假设输入序列长度为24，经过2次池化后长度变为24/2/2=6
        self.fc1 = nn.Linear(32 * 6, 1)

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 第一层卷积 + 批量归一化 + ReLU + 最大池化
        x = self.relu(self.conv1(x))
        x = self.pool(x)

        # 第二层卷积 + 批量归一化 + ReLU + 最大池化
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # 展平
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc1(x)

        return x

# 初始化模型、损失函数和优化器
model = CNNModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 评估模型
model.eval()
predictions = []
actual = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        predictions.extend(outputs.numpy().flatten())
        actual.extend(batch_y.numpy().flatten())



# 计算评估指标
mse = mean_squared_error(actual, predictions)
rmse = np.sqrt(mse)


print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')

# 绘制预测结果
plt.figure(figsize=(12, 6))
plt.plot(actual, label='真实值')
plt.plot(predictions, label='预测值', color='red')
plt.title('CNN模型预测结果')
plt.xlabel('时间')
plt.ylabel('电网负荷 [MW]')
plt.legend()
plt.savefig('cnn_prediction.png', dpi=300)
plt.show()