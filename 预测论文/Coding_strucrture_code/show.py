import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 8  # 设置全局字体大小为 8
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 读取数据集
df = pd.read_csv(r'C:\Users\huahs\PycharmProjects\LSTM-pytorch\预测论文\dataset\YearlyEliaGridLoadByQuarterHour_2022.csv')

# 去除逗号并将电力负荷值转换为浮点数
df['Elia Grid Load [MW]'] = df['Elia Grid Load [MW]'].str.replace(',', '').astype(float)

# 将时间列转换为日期时间格式
df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)'], format='%d/%m/%Y %H:%M:%S')

# 按升序对数据进行排序
df.sort_values('Datetime (UTC)', ascending=True, inplace=True)

# 绘制图表
plt.figure(figsize=(6,5))
plt.plot(df['Elia Grid Load [MW]'][:96*12], linestyle='-', marker=None)
plt.title('Partial Load Information', fontsize=13)  # 标题字体稍大一些
plt.xlabel('Time Point', fontsize=13)
plt.ylabel('Load [MW]', fontsize=13)
plt.tick_params(axis='both', which='major', labelsize=13)  # 设置刻度标签的字体大小
plt.grid(True)
plt.tight_layout()
plt.savefig('../save_pic/00.jpg', dpi=300)
plt.show()