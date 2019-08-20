# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:07:01 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
import datetime
# #定义文件路径
# data_src='./data/'
# data_des='./temp/'

# starttime = "2018/1/4  0:00:00"    endtime="2018/1/5 0:00:00"

def C_zd(starttime, endtime,df2,df1):

    # df1 = pd.DataFrame(data_dz)  # 读取地点文件
    # df2 = pd.DataFrame(data)  # 读取case发生文件
    #print('starttime',starttime)
    #print('endtime',endtime)

    starttime = datetime.datetime.strptime(starttime, "%Y/%m/%d %H:%M:%S")  # 字符串转化为date形式
    endtime = datetime.datetime.strptime(endtime, "%Y/%m/%d %H:%M:%S")  # 字符串转化为date形式

    #print(type(starttime))

    df2 = df2[( pd.to_datetime(df2["date"]) >= starttime) & ( pd.to_datetime(df2["date"]) <= endtime)]  # 限制时间范围  pd.to_datetime转化为日期

    print('df1',df1)
    #  去重复,地点集
    list1 = []
    d1 = list(df1["序号"])
    d1 = set(d1)  # set去重
    for c in d1:  # 将去重的数据集存入列表list1
        list1.append(c)
    list1.sort()    # 列表从小到大排序
    #print('list1',list1)

    #  去重复,时间集
    list2 = []
    d2 = list(df2["date"])
    #print('d2',d2)
    d2 = set(d2)  # 去重

    for c in d2:
        list2.append(c)

    def get_list(date):     # str 转 date 函数
        return datetime.datetime.strptime(date, "%Y/%m/%d %H:%M")

    list2 = sorted(list2, key=lambda date: get_list(date))   #  date 排序
    print('list2',list2)

    # 二维数组 初始化
    arr = []
    for i in range(len(d1)):
        arr.append([])
        for j in range(len(d2)):
            arr[i].append(0)
    #print('arr',arr)

    df2 = df2.groupby(['z_id', 'date'], as_index=False).sum()
    #print('a1', a)
    #a = a['num'].tolist()
   # print('a', a)


    # 赋值
    for i in range(len(d1)):
        for j in range(len(d2)):
            #a = df2["num"].groupby(df2[(df2["z_id"] == list1[i]) & (df2["date"] == list2[j])]).sum()
            list3 = datetime.datetime.strptime(list2[j], "%Y/%m/%d %H:%M")  # 字符串转化为date形式
            #print('list3',list3)
            a = df2[(df2["z_id"] == list1[i]) & ((pd.to_datetime(df2["date"])) == list3 )]["num"].tolist()   # 选取满足条件的值
            # if a != "":
            if a == []:   #  如果没有数据，用0填充
                arr[i][j] = 0
            else:
                arr[i][j] = int(a[0])  # 有数据用 num 填充
    #print('arr',arr)
    arr = np.array(arr)  # list 转数组
    m, n = arr.shape  # m行n列

    # 返回C矩阵
    C = arr
    cc = pd.DataFrame(C)

    #print(cc)
    # C 求和
    C_sum = arr.sum()
   # print(C_sum)
    # C_z 指定区域z上的全段病例数，1表示行
    C_z = arr.sum(axis=1)
   # print('np.mat(C_z).T',np.mat(C_z).T)
   # m, n = arr.shape  # m行n列
   # k, m = np.size(C_z)
    #print('C_z',C_z)

    # C_d 指定时间段d上的全段病例数全区病例数,0表示列
    C_d = arr.sum(axis=0)
    #print('C_d',C_d)
    #print(len(C_d))

    # 計算期望值
    u_zd = np.multiply(C_d,np.mat(C_z).T) / C_sum
    #print('u_zd',u_zd)

    u_zd = np.array(u_zd)  # list 转数组


    return C,C_sum,u_zd,C_z,C_d


# def main():
#     data_file = data_src + r'fenbei_data.csv'
#     data = pd.read_csv(data_file)
#     data_file_dz = data_src + 'city1.csv'
#     data_dz = pd.read_csv(data_file_dz)
#
#     # 开始结束时间
#     starttime = "2018/1/2 0:00:00"
#     endtime = "2018/1/5 0:00:00"
#
#     C_1, C_z1, C_d1, u_zd = C_zd(starttime,endtime,data,data_dz)
#     print(C_1)
#     print(C_z1)
#     print(C_d1)
#     print(u_zd)
#
# main()