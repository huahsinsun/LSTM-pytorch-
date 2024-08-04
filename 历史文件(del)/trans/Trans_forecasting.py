
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from 文件预处理 import *


# 加载数据
def load_data():
    train_data = pd.read_csv(TRAIN_OUTPUT_PATH)
    val_data = pd.read_csv(VAL_OUTPUT_PATH)
    test_data = pd.read_csv(TEST_OUTPUT_PATH)
    return train_data, val_data, test_data

# 准备数据为模型可用的格式
def prepare_data(data, input_window, output_window):
    X, y = [], []
    for i in range(len(data) - input_window - output_window + 1):
        X.append(data[i:(i+input_window)].values)
        y.append(data[(i+input_window):(i+input_window+output_window)].values)
    return np.array(X), np.array(y)

# 创建数据加载器
def create_dataloader(X, y, batch_size):
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead, num_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.input_linear = nn.Linear(input_size, d_model)
        self.output_linear = nn.Linear(d_model, output_size)

    def forward(self, src):
        src = self.input_linear(src)
        output = self.transformer_encoder(src)
        output = self.output_linear(output[:, -1, :])
        return output


# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.view(batch_y.size(0), -1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.view(batch_y.size(0), -1))
                val_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

# 预测
def predict(model, test_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            predictions.append(outputs.cpu().numpy())
    return np.concatenate(predictions)

def main():
    # 设置参数
    input_window = 7  # 输入窗口大小
    output_window = 1  # 输出窗口大小
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    d_model = 64
    nhead = 4
    num_layers = 2
    dim_feedforward = 256

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_data, val_data, test_data = load_data()
    #return train_data, val_data, test_data

    # 准备数据
    X_train, y_train = prepare_data(train_data, input_window, output_window)
    X_val, y_val = prepare_data(val_data, input_window, output_window)
    X_test, y_test = prepare_data(test_data, input_window, output_window)

    # 创建数据加载器
    train_loader = create_dataloader(X_train, y_train, batch_size)
    val_loader = create_dataloader(X_val, y_val, batch_size)
    test_loader = create_dataloader(X_test, y_test, batch_size)

    # 初始化模型
    input_size = X_train.shape[2]
    output_size = y_train.shape[1] * y_train.shape[2]
    model = TransformerModel(input_size, output_size, d_model, nhead, num_layers, dim_feedforward).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # 预测
    predictions = predict(model, test_loader, device)

    # 这里可以添加代码来评估模型性能，例如计算RMSE或MAE
    # 也可以将预测结果保存到文件中

    print("训练和预测完成")

if __name__ == "__main__":
    main()
