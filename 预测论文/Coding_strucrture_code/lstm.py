'''相比于shorten，一些config过时且不合理，比如文件路径等'''
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


#region
# 加载 scaler 参数
data_min, data_max = np.loadtxt('../dataset/scaler_params.txt')
def inverse_transform(scaled_data, data_min, data_max):
    """反归一化数据"""
    return scaled_data * (data_max - data_min) + data_min

TRAIN_OUTPUT_PATH =None
VAL_OUTPUT_PATH = None
TEST_OUTPUT_PATH = None
def loading_data(i):
    global TRAIN_OUTPUT_PATH, VAL_OUTPUT_PATH, TEST_OUTPUT_PATH
    if i == 0:
        TRAIN_OUTPUT_PATH = '../dataset/short_term_train.csv'
        VAL_OUTPUT_PATH = '../dataset/short_term_val.csv'
        TEST_OUTPUT_PATH = '../dataset/short_term_test.csv'
    elif i == 1:
        TRAIN_OUTPUT_PATH = '../dataset/long_term_train.csv'  # 假设这里是不同的路径
        VAL_OUTPUT_PATH = '../dataset/long_term_val.csv'      # 假设这里是不同的路径
        TEST_OUTPUT_PATH = '../dataset/long_term_test.csv'    # 假设这里是不同的路径

loading_data(1)

# 数据加载和准备
train_df = pd.read_csv(TRAIN_OUTPUT_PATH)
val_df = pd.read_csv(VAL_OUTPUT_PATH)
test_df = pd.read_csv(TEST_OUTPUT_PATH)

# 将数据集转换为PyTorch的Tensor
train_data = torch.tensor(train_df['Elia Grid Load [MW]'].values, dtype=torch.float32).unsqueeze(1)
val_data = torch.tensor(val_df['Elia Grid Load [MW]'].values, dtype=torch.float32).unsqueeze(1)
test_data = torch.tensor(test_df['Elia Grid Load [MW]'].values, dtype=torch.float32).unsqueeze(1)


def add_noise(data, noise_level=0):
    noise = torch.randn(data.shape) * noise_level * torch.std(data)
    return data + noise
# 设置随机种子以确保结果可复现

# 添加噪声
noise_level = 0.02  # 可以调整这个值来控制噪声的强度
train_data_noisy = add_noise(train_data, noise_level)
val_data_noisy = add_noise(val_data, noise_level)
test_data_noisy = add_noise(test_data, noise_level)

# r控制数据集大小，可自适应匹配不同数据集，通过此处的硬编码
# r = 1
# train_data = [_ for _ in train_data[:int(len(train_data)*r)]]
# val_data = [_ for _ in val_data[:int(len(val_data)*r)]] #保证训练
# test_data = [_ for _ in test_data[:int(len(test_data)*r)]]
# # 检查训练之前的模式
# plt.figure(figsize=(12, 6))
# plt.plot(test_data[:96])  # 绘制一天的数据
# plt.plot(train_data[:96])
# plt.title('原始测试数据的日负荷曲线')
# plt.xlabel('时间（15分钟间隔）')
# plt.ylabel('负荷 [MW]')
# plt.show()
# 打印张量以确认转换
print("Train data shape:", train_data.shape)
print("Validation data shape:", val_data.shape)
print("Test data shape:", test_data.shape)
#endregion


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


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=4, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 添加一个额外的线性层
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, output_size)
        self.dropout = nn.Dropout(0.3)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = torch.mean(out, dim=1)
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


output_size = 1
num_epochs = 30
learning_rate = 0.001
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


# 训练和评估函数
def train_and_evaluate(seq_length):
    # 创建数据集和数据加载器
    train_dataset = TimeSeriesDataset(train_data, seq_length)
    val_dataset = TimeSeriesDataset(val_data, seq_length)
    test_dataset = TimeSeriesDataset(test_data, seq_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    # 初始化模型、损失函数和优化器
    model = LSTMModel(1,32,4,1) #rate 0.001 epo15 1 64 2 1
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # # 模型训练
    # best_val_loss = float('inf')
    # for epoch in range(num_epochs):
    #     model.train()
    #     for step, (inputs, targets) in enumerate(train_dataloader, 1):
    #         inputs, targets = inputs.to(device), targets.to(device)
    #         optimizer.zero_grad()
    #         outputs = model(inputs)
    #         loss = criterion(outputs, targets)
    #         loss.backward()
    #         optimizer.step()
    #
    #         # 每个epoch打印20次预测信息
    #         if step % (len(train_dataloader) // 5 + 1) == 0:
    #             print(f"Epoch {epoch + 1}/{num_epochs}, 预测步骤 {step}/{len(train_dataloader)}, "
    #                  f"预测值: {outputs[-1].item():.4f}, 实际值: {targets[-1].item():.4f}, abs: {abs(outputs[-1].item() - targets[-1].item()):.4f}")
    #
    #     # 验证过程（保持不变）
    #     model.eval()
    #     val_loss = 0.0
    #     val_predictions, val_true_values = [], []
    #     with torch.no_grad():
    #         for inputs, targets in val_dataloader:
    #             inputs, targets = inputs.to(device), targets.to(device)
    #             outputs = model(inputs)
    #             val_loss += criterion(outputs, targets).item()
    #             val_predictions.extend(outputs.detach().cpu().numpy())
    #             val_true_values.extend(targets.detach().cpu().numpy())
    #
    #     val_loss /= len(val_dataloader)
    #     val_predictions = np.array(val_predictions)
    #     val_true_values = np.array(val_true_values)
    #     # 计算额外的性能指标
    #     val_mse = np.mean((val_predictions - val_true_values) ** 2)
    #     val_rmse = np.sqrt(val_mse)
    #     print(f"Validation Loss: {val_loss:.4f},Loss: {loss.item():.4f} \n MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}")
    #
    #     if val_loss < best_val_loss:
    #         best_val_loss = val_loss
    #         torch.save(model.state_dict(), f'../save_model_para/best_model_seq{seq_length}.pt')
    # # 测试过程（保持不变）
    model.load_state_dict(torch.load(f'../save_model_para/best_model_seq{seq_length}.pt'))
    model.eval()
    predictions, true_values = [], []
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.detach().cpu().numpy())
            true_values.extend(targets.detach().cpu().numpy())
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



seq_lengths = [24]
results = [train_and_evaluate(sl) for sl in seq_lengths]
import matplotlib
# 设置matplotlib的字体为SimHei
matplotlib.rcParams['font.sans-serif'] = ['SimSun']  # 或者使用其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 结果可视化
plt.figure(figsize=(22,8))
for i, (predictions, true_values, mse, rmse,dtw) in enumerate(results):
    plt.subplot(2,2,i + 1)
    plt.plot(true_values[:96], label='真实值')
    plt.plot(predictions[:96], label='预测值')
    # 设置刻度和标签
    # plt.xticks(ticks=np.arange(0, 96, 16), labels=[f'{j // 4:02d}:00' for j in range(0, 96, 16)])
    plt.title(f'序列长度 {seq_lengths[i]} ')
    plt.xlabel('时间')
    plt.ylabel('负荷 [MW]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../save_pic/{seq_lengths[i]}.jpg', dpi=350)


# 性能指标输出
for i, (predictions, true_values, mse, rmse, dtw) in enumerate(results):
    print(f'序列长度 {seq_lengths[i]} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, DTW: {dtw:.4f}')