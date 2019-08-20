import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data1 = r".\temp\39\outbreak_3.15-3.21.xlsx"
data2 = r".\temp\39\outbreak_1.15-1.21.xlsx"

# 读取文件
df1 = pd.DataFrame(pd.read_excel(data1))
df2 = pd.DataFrame(pd.read_excel(data2))

# 合并z_id对应的值
df1 = df1.groupby(['z_id'], as_index=False).sum()
df2 = df2.groupby(['z_id'], as_index=False).sum()

#print('df')
#print(df)

def df_process(df):
    # 存储index值，初始化
    index = []
    # 存储value值，初始化
    v = []

    for i in range(1,43):
        index.append(i)
        a = df[(df["z_id"] == i)]["num"].tolist()  # 选取满足条件的值

        if a == []:  # 如果没有数据，用0填充
            v.append(0)
        else:
            v.append(int(a[0]))  # 有数据用 num 填充
    return index,v

index1,v1 = df_process(df1)
index2,v2 = df_process(df2)

# 创建一个点数为 8 x 6 的窗口, 并设置分辨率为 80像素/每英寸
plt.figure(figsize=(8, 6), dpi=80)

plt.subplot(211)
# 柱子的宽度
width = 0.5
# 绘制柱状图, 每根柱子的颜色为紫罗兰色
p2 = plt.bar(index2, v2, width, label="case", color="#6495ED")
# 设置横轴标签
plt.xlabel('locations')
# 设置纵轴标签
plt.ylabel('observed case')
# 添加标题
plt.title('1.15-1.21 outbreak case')
# # 添加纵横轴的刻度
plt.xticks(np.arange(0, 42, 2))
# plt.yticks(np.arange(0, 81, 10))
# 添加图例
plt.legend(loc="upper right")

plt.subplots_adjust(hspace=0.4)

plt.subplot(212)
#pl.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4)
# 柱子的宽度
width = 0.5
# 绘制柱状图, 每根柱子的颜色为紫罗兰色
p2 = plt.bar(index1, v1, width, label="case", color="#9F79EE")

# 设置横轴标签
plt.xlabel('locations')
# 设置纵轴标签
plt.ylabel('observed case')

# 添加标题
plt.title('3.8-3.15 outbreak case')

# # 添加纵横轴的刻度
plt.xticks(np.arange(0, 42, 2))
# plt.yticks(np.arange(0, 81, 10))

# 添加图例
plt.legend(loc="upper right")

plt.show()