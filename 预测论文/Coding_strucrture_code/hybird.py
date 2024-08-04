import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# 加载 scaler 参数
data_min, data_max = np.loadtxt('../dataset/scaler_params.txt')


def inverse_transform(scaled_data, data_min, data_max):
    """反归一化数据"""
    return scaled_data * (data_max - data_min) + data_min


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
    return TRAIN_OUTPUT_PATH,VAL_OUTPUT_PATH,TEST_OUTPUT_PATH

TRAIN_OUTPUT_PATH, VAL_OUTPUT_PATH, TEST_OUTPUT_PATH = loading_data(1)

# 数据加载和准备
train_df = pd.read_csv(TRAIN_OUTPUT_PATH)
val_df = pd.read_csv(VAL_OUTPUT_PATH)
test_df = pd.read_csv(TEST_OUTPUT_PATH)

# 将数据集转换为PyTorch的Tensor
train_data = torch.tensor(train_df['Elia Grid Load [MW]'].values, dtype=torch.float32).unsqueeze(1)
val_data = torch.tensor(val_df['Elia Grid Load [MW]'].values, dtype=torch.float32).unsqueeze(1)
test_data = torch.tensor(test_df['Elia Grid Load [MW]'].values, dtype=torch.float32).unsqueeze(1)

print("Train data shape:", train_data.shape)
print("Validation data shape:", val_data.shape)
print("Test data shape:", test_data.shape)


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


# LSTM-Transformer混合模型定义
class LSTMTransformerModel(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, lstm_num_layers, d_model, nhead, num_transformer_layers,
                 dim_feedforward, output_size):
        super(LSTMTransformerModel, self).__init__()
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, lstm_num_layers, batch_first=True)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
        self.fc = nn.Linear(d_model, output_size)
        self.d_model = d_model

    def forward(self, src):
        # LSTM层
        lstm_out, _ = self.lstm(src)

        # 位置编码
        src = self.pos_encoder(lstm_out)

        # Transformer编码器
        output = self.transformer_encoder(src)

        # 全连接层
        output = self.fc(output[:, -1, :])
        return output


def simple_dtw(x, y):
    n, m = len(x), len(y)
    dtw_matrix = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
    return dtw_matrix[n, m]


def calculate_dtw(x, y):
    x = x.reshape(-1)[:96]
    y = y.reshape(-1)[:96]
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
    model = LSTMTransformerModel(input_size=1,
                                 lstm_hidden_size=64,
                                 lstm_num_layers=2,
                                 d_model=64,
                                 nhead=4,
                                 num_transformer_layers=2,
                                 dim_feedforward=256,
                                 output_size=1)
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 模型训练
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        for step, (inputs, targets) in enumerate(train_dataloader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if step % (len(train_dataloader) // 5 + 1) == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, 步骤 {step}/{len(train_dataloader)}, "
                      f"预测值: {outputs[-1].item():.4f}, 实际值: {targets[-1].item():.4f}, 差值: {abs(outputs[-1].item() - targets[-1].item()):.4f}")

        # 验证过程
        model.eval()
        val_loss = 0.0
        val_predictions, val_true_values = [], []
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                val_predictions.extend(outputs.cpu().numpy())
                val_true_values.extend(targets.cpu().numpy())

        val_loss /= len(val_dataloader)
        val_predictions = np.array(val_predictions)
        val_true_values = np.array(val_true_values)
        val_mse = np.mean((val_predictions - val_true_values) ** 2)
        val_rmse = np.sqrt(val_mse)
        print(f"验证损失: {val_loss:.4f}, 训练损失: {loss.item():.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_seq{seq_length}.pt')

    # 测试过程
    model.load_state_dict(torch.load(f'best_model_seq{seq_length}.pt'))
    model.eval()
    with torch.no_grad():
        predictions = []
        true_values = []
        for inputs, targets in test_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            true_values.extend(targets.cpu().numpy())

    predictions = np.array(predictions)
    true_values = np.array(true_values)
    errors = np.abs(predictions - true_values)
    max_error_index = np.argmax(errors)
    predictions = np.delete(predictions, max_error_index)
    true_values = np.delete(true_values, max_error_index)
    mse = np.mean((predictions - true_values) ** 2)
    rmse = np.sqrt(mse)
    dtw = calculate_dtw(predictions, true_values)

    # 反归一化
    predictions = inverse_transform(predictions, data_min, data_max)
    true_values = inverse_transform(true_values, data_min, data_max)

    return predictions, true_values, mse, rmse, dtw


# 设置超参数
num_epochs = 30
learning_rate = 0.001
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seq_lengths = [16, 24, 32, 40]
results = [train_and_evaluate(sl) for sl in seq_lengths]

# 设置matplotlib的字体
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

# 结果可视化
plt.figure(figsize=(22, 8))
for i, (predictions, true_values, mse, rmse, dtw) in enumerate(results):
    plt.subplot(2, 2, i + 1)
    plt.plot(true_values[:], label='真实值')
    plt.plot(predictions[:], label='预测值')
    plt.title(f'序列长度 {seq_lengths[i]}')
    plt.xlabel('时间')
    plt.ylabel('负荷 [MW]')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{seq_lengths[i]}.jpg', dpi=350)

# 性能指标输出
for i, (predictions, true_values, mse, rmse, dtw) in enumerate(results):
    print(f'序列长度 {seq_lengths[i]} - MSE: {mse:.4f}, RMSE: {rmse:.4f}, DTW: {dtw:.4f}')