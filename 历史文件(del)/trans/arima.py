import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller

# 设置matplotlib的字体
matplotlib.rcParams['font.sans-serif'] = ['SimSun']
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据加载
train_df = pd.read_csv(r'C:\Users\huahs\PycharmProjects\LSTM-pytorch\dataset\train_dataset.csv')
test_df = pd.read_csv(r'C:\Users\huahs\PycharmProjects\LSTM-pytorch\dataset\test_dataset.csv')

# 将数据转换为NumPy数组
train_data = train_df['Elia Grid Load [MW]'].values
test_data = test_df['Elia Grid Load [MW]'].values

# 数据预处理函数
def preprocess_data(data):
    # 这里可以添加更多的预处理步骤，如处理异常值、归一化等
    # return np.log1p(data)  # 对数变换，使数据分布更接近正态
    return data

# 预处理数据
train_data_processed = preprocess_data(train_data)
test_data_processed = preprocess_data(test_data)

# 检查平稳性
def check_stationarity(timeseries):
    result = adfuller(timeseries, autolag='AIC')
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:', result[4])
    if result[1] <= 0.05:
        print("数据是平稳的")
    else:
        print("数据不是平稳的，可能需要差分")

check_stationarity(train_data_processed)

# ARIMA模型参数
p, d, q = 1, 1, 1  # 这些参数可以通过网格搜索等方法优化

# 滚动预测函数
def rolling_forecast(train, test, p, d, q, steps):
    history = list(train)
    predictions = []
    for t in range(steps):
        model = ARIMA(history, order=(p,d,q))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print(f'预测步骤 {t+1}/{steps}, 预测值: {yhat}, 实际值: {obs}')
    return predictions

# 执行滚动预测
forecast_steps = 672
forecasts = rolling_forecast(train_data_processed[:36], test_data_processed, p, d, q, forecast_steps)

# # 反转预处理（如果使用了对数变换）
forecasts_original = forecasts
test_data_original = test_data_processed[:forecast_steps]

# 绘制真实值与预测值对比图
plt.figure(figsize=(12, 8))
plt.plot(test_data_original, label='真实值')
plt.plot(forecasts_original, label='预测值', color='red')
plt.title('真实值与ARIMA模型滚动预测对比')
plt.xlabel('时间点')
plt.ylabel('电网负荷 [MW]')
plt.legend()
plt.savefig('arima',dpi=350)
plt.show()

# 计算性能指标
mse = mean_squared_error(test_data_original, forecasts_original)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_data_original, forecasts_original)
mape = np.mean(np.abs((test_data_original - forecasts_original) / test_data_original)) * 100

print(f'MSE: {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAE: {mae:.4f}')
print(f'MAPE: {mape:.4f}%')