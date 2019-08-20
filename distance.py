# 计算两个经纬度间的距离

import math
from numpy import mat

# 3.6378.137为地球半径，单位为千米；
EARTH_REDIUS = 6378.137

#定义圆周计算方法
def rad(d):
    #print('d',d)
    #print(type(d))
    rad = float(d) * math.pi / 180.0
    #print(rad)
    return rad

#定义两个经纬度之间的距离（采用谷歌地图距离计算方法）
def getDistance(lat1, lng1, lat2, lng2):      #  Lat1 Lung1 表示A点经纬度，Lat2 Lung2 表示B点经纬度
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    #print(a)
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a/2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b/2), 2)))
    s = s * EARTH_REDIUS
    return s

#距离矩阵L生成函数
# def L_Distance(D,data_dz):
#     D_data = data_dz
#     L = []
#     for i in range(0,D):
#         L.append([])
#         for j in range(0, D):
#             L[i].append(getDistance(D_data['经度'].ix[i], D_data['纬度'].ix[i], D_data['经度'].ix[j], D_data['纬度'].ix[j]))
#     return mat(L).T

# = getDistance(111.53536,23.23464,116.41637,39.92855)
#a = getDistance(116.68221,23.3535,113.26436,23.12908)

#print(a)