import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# 数据路径
LOAD_PATH = r'C:\Users\huahs\PycharmProjects\LSTM-pytorch\dataset\load.CSV'
WEATHER_PATH = r'C:\Users\huahs\PycharmProjects\LSTM-pytorch\dataset\weather.CSV'

# 数据划分比例
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

# 输出路径
TRAIN_OUTPUT_PATH = '../dataset/train_dataset.csv'
VAL_OUTPUT_PATH = '../dataset/val_dataset.csv'
TEST_OUTPUT_PATH = '../dataset/test_dataset.csv'
OUTPUT_PATH = '../dataset/neo.csv'
# data_processing.py



def load_and_preprocess_data():
    # 文件读取
    load = pd.read_csv(LOAD_PATH)
    weather = pd.read_csv(WEATHER_PATH)

    # 删除不需要的列并保存列名
    load.drop('YMD', axis=1, inplace=True)
    load_columns = load.columns
    weather_columns = weather.columns

    # 处理缺失值，例如填充
    weather.fillna(method='ffill', inplace=True)

    # 归一化
    scaler = MinMaxScaler()
    load_normalized = scaler.fit_transform(load.T).T
    weather_normalized = scaler.fit_transform(weather.T).T

    # 按行进行归一化
    # 将 numpy 数组转换为 DataFrame，使用原始列名
    load_df = pd.DataFrame(load_normalized[:1113][:], columns=load_columns)
    weather_df = pd.DataFrame(weather_normalized, columns=weather_columns)

    load_df = load_df.T
    weather_df = weather_df.T
    print(load_df)

    neo = pd.concat([load_df, weather_df], axis=0)
    neo.to_csv(OUTPUT_PATH)

    return neo

def split_and_save_data(data):
    # 将数据集划分为训练集、验证集和测试集
    train_data, test_data = train_test_split(data, test_size=TEST_RATIO, shuffle=False)
    _, val_data = train_test_split(data, test_size=VAL_RATIO/(TRAIN_RATIO+VAL_RATIO), shuffle=False)

    # 保存划分后的数据集为CSV文件
    train_data.to_csv(TRAIN_OUTPUT_PATH, index=False)
    val_data.to_csv(VAL_OUTPUT_PATH, index=False)
    test_data.to_csv(TEST_OUTPUT_PATH, index=False)
    print('数据划分已完成')


data = load_and_preprocess_data()
split_and_save_data(data)
print(data)

