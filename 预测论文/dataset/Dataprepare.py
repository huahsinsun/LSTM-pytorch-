import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 读取数据集
df = pd.read_csv(
    r'C:\Users\huahs\PycharmProjects\LSTM-pytorch\预测论文\dataset\YearlyEliaGridLoadByQuarterHour_2022.csv')

# 处理缺失值
df.fillna(method='ffill', inplace=True)  # 使用前向填充方法填充缺失值

# 去除逗号并将电力负荷值转换为浮点数
df['Elia Grid Load [MW]'] = df['Elia Grid Load [MW]'].str.replace(',', '').astype(float)

# 将时间列转换为日期时间格式
df['Datetime (CET+1/CEST +2)'] = pd.to_datetime(df['Datetime (CET+1/CEST +2)'], format='%d/%m/%Y %H:%M:%S')
df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'], format='%d/%m/%Y %H:%M:%S')

# 按升序对数据进行排序
df.sort_values('Datetime (UTC)', ascending=True, inplace=True)

# 数据归一化
scaler = MinMaxScaler()
df['Elia Grid Load [MW]'] = scaler.fit_transform(df[['Elia Grid Load [MW]']])

# 定义天数
SHORT_TERM_TOTAL = 45
SHORT_TERM_TRAIN = 30
SHORT_TERM_VAL = 10
SHORT_TERM_TEST = 5

LONG_TERM_TRAIN = 240  # 约8个月
LONG_TERM_VAL = 60  # 约2个月
LONG_TERM_TEST = 60  # 约2个月


def days_to_rows(days):
    return days * 24 * 4  # 每天24小时，每小时4个15分钟间隔


def split_data(df, is_short_term=True):
    if is_short_term:
        # 短期预测
        total_rows = days_to_rows(SHORT_TERM_TOTAL)
        df = df.tail(total_rows)
        train_data = df.head(days_to_rows(SHORT_TERM_TRAIN))
        val_data = df.tail(days_to_rows(SHORT_TERM_VAL + SHORT_TERM_TEST)).head(days_to_rows(SHORT_TERM_VAL))
        test_data = df.tail(days_to_rows(SHORT_TERM_TEST))
    else:
        # 长期预测
        train_data = df.head(days_to_rows(LONG_TERM_TRAIN))
        val_data = df.tail(days_to_rows(LONG_TERM_VAL + LONG_TERM_TEST)).head(days_to_rows(LONG_TERM_VAL))
        test_data = df.tail(days_to_rows(LONG_TERM_TEST))

    return train_data, val_data, test_data


# 短期预测数据划分
train_short, val_short, test_short = split_data(df, is_short_term=True)

# 长期预测数据划分
train_long, val_long, test_long = split_data(df, is_short_term=False)

# 保存短期预测数据集
train_short.to_csv('../dataset/short_term_train.csv', index=False)
val_short.to_csv('../dataset/short_term_val.csv', index=False)
test_short.to_csv('../dataset/short_term_test.csv', index=False)

# 保存长期预测数据集
train_long.to_csv('../dataset/long_term_train.csv', index=False)
val_long.to_csv('../dataset/long_term_val.csv', index=False)
test_long.to_csv('../dataset/long_term_test.csv', index=False)

print('数据划分已完成，CSV文件已保存。')

# 绘制短期预测数据划分图
plt.figure(figsize=(12, 6))
plt.plot(train_short['Datetime (UTC)'], train_short['Elia Grid Load [MW]'], label='Train')
plt.plot(val_short['Datetime (UTC)'], val_short['Elia Grid Load [MW]'], label='Validation')
plt.plot(test_short['Datetime (UTC)'], test_short['Elia Grid Load [MW]'], label='Test')
plt.title('Short-term Prediction Data Split')
plt.xlabel('Date')
plt.ylabel('Normalized Load')
plt.legend()
plt.savefig('../dataset/short_term_split.png')
plt.close()

# 绘制长期预测数据划分图
plt.figure(figsize=(12, 6))
plt.plot(train_long['Datetime (UTC)'], train_long['Elia Grid Load [MW]'], label='Train')
plt.plot(val_long['Datetime (UTC)'], val_long['Elia Grid Load [MW]'], label='Validation')
plt.plot(test_long['Datetime (UTC)'], test_long['Elia Grid Load [MW]'], label='Test')
plt.title('Long-term Prediction Data Split')
plt.xlabel('Date')
plt.ylabel('Normalized Load')
plt.legend()
plt.savefig('../dataset/long_term_split.png')
plt.close()

print('数据划分图已保存。')