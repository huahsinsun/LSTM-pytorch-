import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 读取数据集
df = pd.read_csv(r'C:\Users\huahs\PycharmProjects\LSTM-pytorch\预测论文\dataset\YearlyEliaGridLoadByQuarterHour_2022.csv')


# 处理缺失值
df.fillna(method='ffill', inplace=True)  # 使用前向填充方法填充缺失值

# 去除逗号并将电力负荷值转换为浮点数
df['Elia Grid Load [MW]'] = df['Elia Grid Load [MW]'].str.replace(',', '.').astype(float)

# 将时间列转换为日期时间格式
df['Datetime (CET+1/CEST +2)'] = pd.to_datetime(df['Datetime (CET+1/CEST +2)'], format='%d/%m/%Y %H:%M:%S')
df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'], format='%d/%m/%Y %H:%M:%S')

# 按升序对数据进行排序
df.sort_values('Datetime (UTC)', ascending=True, inplace=True)

# 只保留 Elia Grid Load 列
df = df[['Elia Grid Load [MW]']]

# 数据归一化
scaler = MinMaxScaler()
df['Elia Grid Load [MW]'] = scaler.fit_transform(df[['Elia Grid Load [MW]']])

# 保存处理后的数据集
df.to_csv('../dataset/processed_dataset.csv', index=False)
print('Data Preparation Complete')


# 数据划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# 输出路径
TRAIN_OUTPUT_PATH = '../dataset/train_dataset.csv'
VAL_OUTPUT_PATH = '../dataset/val_dataset.csv'
TEST_OUTPUT_PATH = '../dataset/test_dataset.csv'
OUTPUT_PATH = '../dataset/neo.csv'

# 将数据集划分为训练集、验证集和测试集
train_data, test_data = train_test_split(data, test_size=TEST_RATIO, shuffle=False)
_, val_data = train_test_split(data, test_size=VAL_RATIO / (TRAIN_RATIO + VAL_RATIO), shuffle=False)

# 保存划分后的数据集为CSV文件
train_data.to_csv(TRAIN_OUTPUT_PATH, index=False)
val_data.to_csv(VAL_OUTPUT_PATH, index=False)
test_data.to_csv(TEST_OUTPUT_PATH, index=False)
print('数据划分已完成')