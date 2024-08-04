import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# 设置matplotlib的字体
matplotlib.rcParams['font.sans-serif'] = ['SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据加载
train_df = pd.read_csv('../dataset/long_term_train.csv')
test_df = pd.read_csv('../dataset/long_term_test.csv')

# 将数据转换为NumPy数组
train_data = train_df['Elia Grid Load [MW]'].values
test_data = test_df['Elia Grid Load [MW]'].values


def create_features(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)


def rolling_svm_forecast(train, test, lookback, steps):
    history = list(train)
    predictions = []
    scaler = StandardScaler()

    for t in range(steps):
        X, y = create_features(history, lookback)
        X_scaled = scaler.fit_transform(X)

        model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        model.fit(X_scaled, y)

        X_pred = np.array(history[-lookback:]).reshape(1, -1)
        X_pred_scaled = scaler.transform(X_pred)
        yhat = model.predict(X_pred_scaled)[0]
        predictions.append(yhat)

        obs = test[t]
        history.append(obs)
        print(f'预测步骤 {t + 1}/{steps}, 预测值: {yhat:.4f}, 实际值: {obs:.4f}')

    return np.array(predictions)


# 添加简单的DTW实现
def simple_dtw(x, y):
    n, m = len(x), len(y)
    dtw_matrix = np.zeros((n + 1, m + 1))
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i - 1] - y[j - 1])
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1])
    return dtw_matrix[n, m]


# 设置参数
lookback = 24  # 使用过去24个时间步的数据进行预测
forecast_steps = 96  # 预测一天（96个15分钟间隔）

# 执行滚动SVM预测
train_data_used = train_data[-5 * 96:]  # 使用最后5天的数据作为初始训练集
forecasts = rolling_svm_forecast(train_data_used, test_data, lookback, forecast_steps)

# 计算性能指标
mse = mean_squared_error(test_data[:forecast_steps], forecasts)
rmse = np.sqrt(mse)
dtw = simple_dtw(test_data[:forecast_steps], forecasts)

# 绘制真实值与预测值对比图
plt.figure(figsize=(12, 8))
plt.plot(test_data[:forecast_steps], label='真实值')
plt.plot(forecasts, label='预测值', color='red')
plt.title('真实值与滚动SVM预测对比')
plt.xlabel('时间')
plt.ylabel('电网负荷 [MW]')
plt.legend()
plt.savefig('../save_pic/rolling_svm_prediction.jpg', dpi=350)
plt.show()

print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'DTW: {dtw:.4f}')