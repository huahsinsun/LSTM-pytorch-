import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

import matplotlib

# 设置matplotlib的字体为SimHei
matplotlib.rcParams['font.sans-serif'] = ['SimSun']  # 或者使用其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

data_min, data_max = np.loadtxt('../dataset/scaler_params.txt')


def inverse_transform(scaled_data, data_min, data_max):
    """反归一化数据"""
    return scaled_data * (data_max - data_min) + data_min


# 加载数据
def load_data(file_path):
    df = pd.read_csv(file_path)
    return torch.tensor(df['Elia Grid Load [MW]'].values, dtype=torch.float32).unsqueeze(1)


TRAIN_OUTPUT_PATH = '../dataset/short_term_train.csv'
VAL_OUTPUT_PATH = '../dataset/short_term_val.csv'
TEST_OUTPUT_PATH = '../dataset/short_term_test.csv'

train_data = load_data(TRAIN_OUTPUT_PATH)
val_data = load_data(VAL_OUTPUT_PATH)
test_data = load_data(TEST_OUTPUT_PATH)


# 数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        return self.data[index:index + self.seq_length], self.data[index + self.seq_length]


# 简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # 调整维度顺序
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.mean(dim=2)  # 全局平均池化
        return self.fc(x)


# 训练函数
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


# 评估函数
def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions, true_values = [], []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy().flatten())
            true_values.extend(targets.cpu().numpy().flatten())

    # 将列表转换为 NumPy 数组
    predictions = np.array(predictions)
    true_values = np.array(true_values)

    mse = np.mean((predictions - true_values) ** 2)
    rmse = np.sqrt(mse)
    dtw = calculate_dtw(predictions, true_values)

    # 反归一化
    predictions = inverse_transform(predictions, data_min, data_max)
    true_values = inverse_transform(true_values, data_min, data_max)

    return predictions, true_values, mse, rmse, dtw




def simple_dtw(x, y):
    n, m = len(x), len(y)
    dtw_matrix = np.zeros((n+1, m+1))
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(x[i-1] - y[j-1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
    return dtw_matrix[n, m]

def calculate_dtw(x, y):
    x = x.reshape(-1)  # 确保x是一维的
    y = y.reshape(-1)  # 确保y是一维的
    x = x[:96]
    y = y[:96]
    return simple_dtw(x, y)
# 主函数
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seq_length = 24
    batch_size = 32
    num_epochs = 30
    learning_rate = 0.001

    train_dataset = TimeSeriesDataset(train_data, seq_length)
    val_dataset = TimeSeriesDataset(val_data, seq_length)
    test_dataset = TimeSeriesDataset(test_data, seq_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = SimpleCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)

        _, _, val_mse, val_rmse , dwt = evaluate(model, val_loader, criterion, device)
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val MSE: {val_mse:.4f}, Val RMSE: {val_rmse:.4f}')

    predictions, true_values, test_mse, test_rmse,dtw = evaluate(model, test_loader, criterion, device)
    print(f'Test MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}')
    print("DTW:{}".format(dtw))

    # 绘制结果
    plt.figure(figsize=(12, 6))
    plt.plot(true_values[:96], label='真实值')
    plt.plot(predictions[:96], label='预测值')
    plt.title('CNN模型预测结果')
    plt.xlabel('时间')
    plt.ylabel('负荷 [MW]')
    plt.legend()
    plt.savefig('../save_pic/cnn_prediction.jpg', dpi=300)
    plt.show()


if __name__ == '__main__':
    main()