import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# 设置matplotlib的字体为SimHei
matplotlib.rcParams['font.sans-serif'] = ['SimSun']  # 或者使用其他支持中文的字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据集
df = pd.read_csv(
    r'C:\Users\huahs\PycharmProjects\LSTM-pytorch\预测论文\dataset\YearlyEliaGridLoadByQuarterHour_2022.csv')

# 去除逗号并将电力负荷值转换为浮点数
df['Elia Grid Load [MW]'] = df['Elia Grid Load [MW]'].str.replace(',', '').astype(float)

# 将时间列转换为日期时间格式
df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'], format='%d/%m/%Y %H:%M:%S')

# 按升序对数据进行排序
df.sort_values('Datetime (UTC)', ascending=True, inplace=True)




# 绘制图表
plt.figure(figsize=(10, 5))
plt.plot(df['Elia Grid Load [MW]'][:96*24])
plt.title(f'部分负荷数据 ')
plt.xlabel('时间点')
plt.ylabel('负荷 [MW]')
plt.grid(True)
plt.tight_layout()
plt.savefig(r'C:\Users\huahs\PycharmProjects\LSTM-pytorch\预测论文\save_pic\pic_view.jpg',dpi=330)
plt.show()