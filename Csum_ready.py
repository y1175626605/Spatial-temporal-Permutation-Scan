# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:07:01 2019

@author: Administrator
"""

import pandas as pd
import numpy as np
import datetime
# #Definition file path
# data_src='./data/'
# data_des='./temp/'

# starttime = "2018/1/4  0:00:00"    endtime="2018/1/5 0:00:00"

def C_zd(starttime, endtime,df2,df1):

    # df1 = pd.DataFrame(data_dz)  # Read location file
    # df2 = pd.DataFrame(data)  # Read case file
    #print('starttime',starttime)
    #print('endtime',endtime)

    starttime = datetime.datetime.strptime(starttime, "%Y/%m/%d %H:%M:%S")  # String converted to date form
    endtime = datetime.datetime.strptime(endtime, "%Y/%m/%d %H:%M:%S")  # String converted to date form

    #print(type(starttime))

    df2 = df2[( pd.to_datetime(df2["date"]) >= starttime) & ( pd.to_datetime(df2["date"]) <= endtime)]  # Limit time range pd.to_datetime is converted to date

    print('df1',df1)
    #  Deduplication, location set
    list1 = []
    d1 = list(df1["序号"])
    d1 = set(d1)  # set Deduplication
    for c in d1:  # Save the deduplicated data set to the list1
        list1.append(c)
    list1.sort()    # List sorted from small to large
    #print('list1',list1)

    #  Deduplication, time set
    list2 = []
    d2 = list(df2["date"])
    #print('d2',d2)
    d2 = set(d2)  # Deduplication

    for c in d2:
        list2.append(c)

    def get_list(date):     # Str to date function
        return datetime.datetime.strptime(date, "%Y/%m/%d %H:%M")

    list2 = sorted(list2, key=lambda date: get_list(date))   #  Date sort
    print('list2',list2)

    # Two-dimensional array initialization
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


    # Assignment
    for i in range(len(d1)):
        for j in range(len(d2)):
            #a = df2["num"].groupby(df2[(df2["z_id"] == list1[i]) & (df2["date"] == list2[j])]).sum()
            list3 = datetime.datetime.strptime(list2[j], "%Y/%m/%d %H:%M")  # String converted to date form
            #print('list3',list3)
            a = df2[(df2["z_id"] == list1[i]) & ((pd.to_datetime(df2["date"])) == list3 )]["num"].tolist()   # Select a value that satisfies the condition
            # if a != "":
            if a == []:   # If there is no data, fill it with 0
                arr[i][j] = 0
            else:
                arr[i][j] = int(a[0])  # Have data filled with num
    #print('arr',arr)
    arr = np.array(arr)  # List to array
    m, n = arr.shape  # m rows and n columns

    # Return to C matrix
    C = arr
    cc = pd.DataFrame(C)

    #print(cc)
    # C Summation
    C_sum = arr.sum()
   # print(C_sum)
    # C_z Specify the total number of events on the area z, 1 indicates the line
    C_z = arr.sum(axis=1)
   # print('np.mat(C_z).T',np.mat(C_z).T)
   # m, n = arr.shape  # m rows and n columns
   # k, m = np.size(C_z)
    #print('C_z',C_z)

    # C_d The total number of events in the specified time period d is the number of events in the whole zone, and 0 is the column.
    C_d = arr.sum(axis=0)
    #print('C_d',C_d)
    #print(len(C_d))

    # Calculate the expected value
    u_zd = np.multiply(C_d,np.mat(C_z).T) / C_sum
    #print('u_zd',u_zd)

    u_zd = np.array(u_zd)  # List to array


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
