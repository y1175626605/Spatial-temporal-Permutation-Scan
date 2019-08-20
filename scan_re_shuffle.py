import pandas as pd
import distance
import Csum_ready
import copy
import math
import operator
import time
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

np.set_printoptions(threshold=np.nan)

pd.set_option('display.width', 1000)  # 设置字符显示宽度
pd.set_option('display.max_rows', None)  # 设置显示最大行
pd.set_option('display.max_columns', None)  # 设置显示最大行


def space_cluster(df1, R_max):
    d = []  # 距离计算

    # circle = []
    # 获取满足条件的经纬度
    def JD(k):  # 定义经度
        # print(df1.序号)
        a = df1[df1.序号 == k].经度
        return a.tolist()[0]

    def WD(k):  # 定义纬度
        b = df1[df1.序号 == k].纬度
        # print(b)
        return b.tolist()[0]

    m, n = df1.shape
    print('m', 'n', m, n)
    # 计算空间聚集区域
    for i in range(m):
        d.append([])
        for j in range(m):
            # print(JD(i+1),WD(i+1),JD(j+1),WD(j+1))
            d[i].append(distance.getDistance(JD(i + 1), WD(i + 1), JD(j + 1), WD(j + 1)))  # 计算当前地点与别的地址的距离
        # print('i',i,','"d[i]",d )
        # print(d)

        for k in range(m):
            # print(d[i][k])
            if (d[i][k] <= R_max):  # 当两个地点的距离小于最大半径的时候，以i为圆心，k为半径画圆
                c = {}
                c["center_xbid"] = i + 1
                c["r"] = d[i][k]
                xbid = []  # 数组存储下标

                # print(d[i][k])
                for v in range(m):  # 当以i为圆心，k为半径画圆, 判断在圆内的地点，存储下标集合
                    # print(d[i][v])
                    if (d[i][k] >= d[i][v]):
                        if (i != k):
                            xbid.append(v + 1)  # 空间聚类集合
                        else:
                            xbid.append(i + 1)  # 只存储圆心
                    else:
                        continue

                #if i == 170:
                   # print('xbid-171', xbid)
                c["xbids"] = xbid
                circle.append(c)  # 存储形式 [{ }{ }{ }]
            else:
                continue
    # print(circle)
    print('d', d)
    return circle


# print(circle)
# print(circle)
def scan(case, circle, zdsjc, U, topN, day):  # 扫描结果计算函数

    C = copy.deepcopy(case)  # 事件矩阵
    circle = copy.deepcopy(circle)  # 中心半径以及空间簇记录列表
    T_max = copy.deepcopy(zdsjc)  # 最大时间簇
    U = copy.deepcopy(U)  # 期望矩阵
    # print('c',C)
    # print('type(U)',type(U))
    # day = 7
    cir = []
    # cir = {}
    for T_cluster in range(0, T_max):  #
        # print("T_cluster")
        # print(T_cluster)
        circle1 = copy.deepcopy(circle)
        for i in range(len(circle1)):
            circle1[i]["T_cluster"] = T_cluster + 1  # 时间簇
            circle1[i]["C_zd"] = ""
            circle1[i]["u_zd"] = ""
            C_zd = 0
            u_zd = 0

            for k in circle1[i]["xbids"]:
                # print(i)
                k = k - 1
                j = day - 1
                while j >= day - 1 - T_cluster:
                    # print(j)
                    C_zd += C[k][j]
                    u_zd += U[k][j]
                    j -= 1
            u_zd = float(u_zd)
            # print("C_zd:{}".format(C_zd))
            if C_zd <= u_zd:
                LGLR = 0  # 实际病例小于预期，正常不警告
            else:
                LGLR = C_zd * math.log(C_zd / u_zd) + (C_sum - C_zd) * math.log(
                    (C_sum - C_zd) / (C_sum - u_zd))  # 计算似然比
            circle1[i]["LGLR"] = LGLR  # circle1[i]中添加似然比key
            circle1[i]["C_zd"] = C_zd
            circle1[i]["u_zd"] = u_zd
        # a = "cir"+ str(T_cluster)

        # print(T_cluster)
        # cir[T_cluster] = circle1
        # print(cir)
        # print(type(circle))
        # print("-----------------------------------")
        # (T_cluster)
        # print('circle1',circle1)
        cir = cir + circle1
    # print(cir)
    # print(cir)
    # T_cluster += 1
    # print(cir)
    # sorted_cir = sorted(cir, key=operator.itemgetter("LGLR"),reverse=True)   # 先对LGLR从大到小排序

    ### 转 df --》 -- 》 去除r = 0 ,LGLR = 0 --》 排序 --》 同值去重 --》 转list
    df_cir = pd.DataFrame(cir)

    # 首先根据 LGLR排序，然后再按照r进行排序
    df_cir = df_cir.sort_index(axis=0, ascending=[False, True], by=['LGLR', 'r'])

    # 删除 LGLR  r 为 0 的行
    ### df_cir = df_cir[~df_cir['LGLR'].isin([0])]  # 会产生空集
    ### df_cir = df_cir[~df_cir['r'].isin([0])]
    # 去重，重复的只取 第一个
    #   df_cir = df_cir.drop_duplicates(subset=['LGLR'],keep='first')    #  似然比从大到小排序后，相等的似然比其半径从小到大排序，留下第一个删除剩下的
    #   df_cir = df_cir.drop_duplicates(subset=['r'], keep='first')
    #   df_cir = df_cir.reset_index(drop=True)

    ## 去重,去除第一个聚集圈内包含的点
    f2 = df_cir.iloc[[0]]
    xb = f2.center_xbid.tolist()

    list1 = f2.xbids.tolist()
    list = copy.deepcopy(list1)
    # list[0].remove(xb[0])
    #f = 0
    # print('f1', f1)
    # print('list',list[0][1])
    # print(f1.iterrows())
    #if list[0][0] == xb[0]:
        #list[0].insert(0,xb[0])

   # print(' xb[0]', xb[0])
    print('list[0]',list[0])

    for i in list[0]:
    #for i in [55,46]:
        f = 0
        for index, row in df_cir.iterrows():
            # print('row',row['name'])
            # print('row',row['center_xbid'])
            # print('xb[0]', xb[0])
            # print('f',f)
            if row['center_xbid'] == xb[0] and f == 0:
                f = 1
               # print('f=1')
            elif row['center_xbid'] == xb[0] and f == 1:
                #print('f=0')
                df_cir.drop(index, axis=0, inplace=True)
            elif i == row['center_xbid']:
                df_cir.drop(index, axis=0, inplace=True)


    df_cir = df_cir.drop_duplicates(subset=['LGLR'], keep='first')  # 似然比从大到小排序后，相等的似然比其半径从小到大排序，留下第一个删除剩下的
    # = df_cir.drop_duplicates(subset=['r'], keep='first')
    df_cir = df_cir.reset_index(drop=True)

    # 取10个元素
    df_cir = df_cir.iloc[0:10]
    # print('df_cir',df_cir)
    # df_cir = df_cir.tolist()
    # df_cir = np.array(df_cir)  # np.ndarray()
    # df_cir = df_cir.tolist()  # list
    # print(df_cir)
    return df_cir
    # 去重


# df_cir = df_cir.drop_duplicates(['L'])
# print(df_cir)


# sorted_cir = sorted(cir,key = lambda  r: r['LGLR'],reverse=True)
# print(sorted_cir)
# sorted_cir = sorted(sorted_cir,key = operator.itemgetter("r"))  # 接着按照r从小到大排序
# print(sorted_cir)
# print("sorted_cir")
# print(sorted_cir)
# print(sorted_cir)
# if topN < len(sorted_cir):
#     sorted_cir = sorted_cir[:topN]   # 取排序后数组的前topN个元素
# return sorted_cir

# Knuth-Durstenfeld Shuffle   算法
def randpermBySYB(p1):
    p = copy.copy(p1)
    p = np.array(p)
    # print(type(p))
    # print(p.shape[0])
    n = np.size(p)
    # print(n)
    # if np.size(p,0)==1:    # 判断是行矩阵
    for i in range(n):
        # print('i',i)
        j = np.random.randint(n - i)  # 产生1到n-1范围内伪随机整数
        # print('j',j)
        tmp = p[i + j]
        p[i + j] = p[i]
        p[i] = tmp  # 交换数据
    return p


## 随机重排
def shuffle(v):
    v = np.array(v)
    # print(v[0])
    m, n = v.shape
    # print('m,',m)
    for i in range(m):
        # print('v[i]',v[i])
        v[i] = randpermBySYB(v[i])
    return v

'''
def shoot(lon, lat, azimuth, maxdist=None):
    """Shooter Function
    Original javascript on http://williams.best.vwh.net/gccalc.htm
    Translated to python by Thomas Lecocq
    """
    glat1 = lat * np.pi / 180.
    glon1 = lon * np.pi / 180.
    s = maxdist / 1.852
    faz = azimuth * np.pi / 180.

    EPS = 0.00000000005
    #if ((np.abs(np.cos(glat1)) < EPS) and not (np.abs(np.sin(faz)) < EPS)):
        #alert("Only N-S courses are meaningful, starting at a pole!")

    a = 6378.13 / 1.852
    f = 1 / 298.257223563
    r = 1 - f
    tu = r * np.tan(glat1)
    sf = np.sin(faz)
    cf = np.cos(faz)
    if (cf == 0):
        b = 0.
    else:
        b = 2. * np.arctan2(tu, cf)

    cu = 1. / np.sqrt(1 + tu * tu)
    su = tu * cu
    sa = cu * sf
    c2a = 1 - sa * sa
    x = 1. + np.sqrt(1. + c2a * (1. / (r * r) - 1.))
    x = (x - 2.) / x
    c = 1. - x
    c = (x * x / 4. + 1.) / c
    d = (0.375 * x * x - 1.) * x
    tu = s / (r * a * c)
    y = tu
    c = y + 1
    while (np.abs(y - c) > EPS):
        sy = np.sin(y)
        cy = np.cos(y)
        cz = np.cos(b + y)
        e = 2. * cz * cz - 1.
        c = y
        x = e * cy
        y = e + e - 1.
        y = (((sy * sy * 4. - 3.) * y * cz * d / 6. + x) *d / 4. - cz) * sy * d + tu

    b = cu * cy * cf - su * sy
    c = r * np.sqrt(sa * sa + b * b)
    d = su * cy + cu * sy * cf
    glat2 = (np.arctan2(d, c) + np.pi) % (2 * np.pi) - np.pi
    c = cu * cy - su * sy * cf
    x = np.arctan2(sy * sf, c)
    c = ((-3. * c2a + 4.) * f + 4.) * c2a * f / 16.
    d = ((e * cy * c + cz) * sy * c + y) * sa
    glon2 = ((glon1 + x - (1. - c) * d * f + np.pi) % (2 * np.pi)) - np.pi

    baz = (np.arctan2(sa, b) + np.pi) % (2 * np.pi)

    glon2 *= 180. / np.pi
    glat2 *= 180. / np.pi
    baz *= 180. / np.pi

    return (glon2, glat2, baz)

def equi(m, centerlon, centerlat, radius, *args, **kwargs):
    glon1 = centerlon
    glat1 = centerlat
    X = []
    Y = []
    for azimuth in range(0, 360):
        glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius)
        X.append(glon2)
        Y.append(glat2)
    X.append(X[0])
    Y.append(Y[0])

    # ~ m.plot(X,Y,**kwargs) #Should work, but doesn't...
    X, Y = m(X, Y)
    plt.plot(X, Y, **kwargs)

def DrawPointMap(m,centerlon, centerlat, radius,*args, **kwargs):
    glon1 = centerlon
    glat1 = centerlat
    X = []
    Y = []
    for azimuth in range(0, 360):
        glon2, glat2, baz = shoot(glon1, glat1, azimuth, radius)
        X.append(glon2)
        Y.append(glat2)
    X.append(X[0])
    Y.append(Y[0])

    # ~ m.plot(X,Y,**kwargs) #Should work, but doesn't...
    X, Y = m(X, Y)
    plt.plot(X, Y, **kwargs)
'''
if __name__ == '__main__':
    #  地址文件地址
    # data_dz = r"data\city_new.csv"
    #  案例数文件地址
    # data_case = r"data\fenbei_data_new.csv"

    data_dz = r"data\city_jjs.csv"
    data_case = r"data\jjs_text2.csv"

    # 读取地点文件
    df1 = pd.DataFrame(pd.read_csv(data_dz))
    # 读取case发生文件
    df2 = pd.DataFrame(pd.read_csv(data_case))

    # print(df2.date)
    # print(xh[0])0
    print(df1)
    print(df2)

    # m 总的参考地址数
    m, n = df1.shape
    # print(m）
    # 初始化
    R_max = 6  # 最大半径
    T_max = 3  # 最大时间簇
    # day = 7    # 参考天数
    topN = 10  # 取GLR前10
    # xbid = []         # 存储空间簇下标
    space_xbis = []  # 存储所有空间簇
    # c = {}           # 字典存放画圆数据，圆心、半径、圆内空间聚集包含地址
    circle = []  # 存储每个扫描窗口的圆心、半径和空间下标聚合集

    starttime = "2018/2/1 0:00:00"  # 开始时间
    endtime = "2018/2/7 23:59:59"  # 结束时间

    # starttime = "2018/1/1 0:00:00"  # 开始时间
    # endtime = "2018/1/5 0:00:00"  # 结束时间
    # starttime = time.mktime(time.strptime(starttime, '%Y-%m-%d %H:%M:%d'))
    # endtime = time.mktime(time.strptime(endtime, '%Y-%m-%d %H:%M:%d'))
    starttime1 = time.mktime(time.strptime(starttime, '%Y/%m/%d %H:%M:%S'))  # time.mktime(tupletime)接受时间元组并返回时间戳
    endtime1 = time.mktime(time.strptime(endtime,'%Y/%m/%d %H:%M:%S'))  # time.strptime(str,fmt='%a %b %d %H:%M:%S %Y') 根据fmt的格式把一个时间字符串解析为时间元组。
    work_days = int((endtime1 - starttime1) / (24 * 60 * 60))  # 毫秒转为天
    day = work_days + 1  # 参考天数
    print('day', day)
    # T_cluster = 0  # 时间簇初始为0
    C, C_sum, U, C_z, C_d = Csum_ready.C_zd(starttime, endtime, df2, df1)  # 计算 C,C_sum, u
    # 距离判断后的空间数组
    circle = space_cluster(df1, R_max)
    print('C', C)
    # print('C_sum',C_sum)
    #  print('C_d',C_d)

    # print('U',U)
    print('circle', circle)

    result_init = scan(C, circle, T_max, U, topN, day)  # 原始数据结果
    print("result_init")
    print(result_init)

    result_init = np.array(result_init)
    result_init = result_init.tolist()
    # print('result_init',result_init)

    fzcs = 999  # 仿真次数
    c_re = np.zeros((1, day))  # 初始化
    c_re = c_re.astype(np.int)
    np.random.seed(0)  # 随机相同v

    # 存储最终结果
    zzjg = []
    # 取原始数据，比较并获取P值
    # for c in result_init:
    count = 0
    fz = []
    # print('c[1]', c[1])
    c = result_init[0]
    for s in range(fzcs):
        c_re = shuffle(C)
        ## 计算期望值
        c_re = np.array(c_re)  # list 转数组
        c_re_sum = c_re.sum()  # C 求和
        c_re_z = c_re.sum(axis=1)  # C_z 指定区域z上的全段病例数，1表示行
        c_re_d = c_re.sum(axis=0)  # C_d 指定时间段d上的全段病例数全区病例数,0表示列
        u_re_zd = np.multiply(c_re_d, np.mat(c_re_z).T) / c_re_sum  # 計算期望值
        u_re_zd = np.array(u_re_zd)  # list 转数组
        # print("蒙特卡洛")
        # 蒙特卡洛扫描
        df_cir = scan(c_re, circle, T_max, u_re_zd, topN, day)
        df_cir = df_cir.iloc[0:1]
        df_cir = np.array(df_cir)
        df_cir = df_cir.tolist()
        # print('df_cir[][]',df_cir[0][1])
        # print('c',c)
        # print('c[0][1]',c[1])
        if df_cir[0][1] > c[1]:
            count += 1
            # print(df_cir[0][1])
        if count > 50:
            break
        else:
            fz.append(df_cir)
    zh = [0] * (len(fz) + 10)
    # 存储最终结果
    a = 0
    for x in result_init:
        zh[a] = x
        a += 1
    print('len(zh)', len(zh))
    # print('zh=',zh[0])
    for i in range(11, len(zh) + 1):
        # print(i)
        zh[i - 1] = fz[i - 11][0]
    print('zh', zh)
    zh = sorted(zh, key=lambda x: x[1], reverse=True)
    # print('zh2',zh)

    for xb in result_init:
        no = zh.index(xb)
        xb.append((no + 1) / len(zh))
        zzjg.append(xb)
        # if no > 50:
        # continue
        # print(zzjg)
        # print(c)

    # 转化为dataFrame 方便查看
    zzjg = pd.DataFrame(zzjg, columns=['C_zd', 'LGLR', 'T_cluster', 'center_xbid', 'r', 'u_zd', 'xbids', 'P-value'])
    print(zzjg)
    data_case = r"temp\smjg_0201_0207_7_10.xlsx"
    zzjg.to_excel(data_case)

    print(zzjg)

    zzjg = zzjg.iloc[0:5]

    '''
    centerlon = []
    centerlat = []
    radius = []

    # 获取聚类中心id，对应寻找其经纬度
    xbs = zzjg.center_xbid
    xbs = xbs.tolist()
    for x in xbs:
        centerlon.append(df1[df1["序号"] == x]["经度"].tolist()[0])
        centerlat.append(df1[df1["序号"] == x]["纬度"].tolist()[0])
    print('centerlon',centerlon)
    print('centerlat',centerlat)

    # 获取聚类半径
    radius = zzjg.r
    radius = radius.tolist()
    print('radius1',radius)

    # map = Basemap(projection='mill',
    #               llcrnrlat=10,
    #               llcrnrlon=72,
    #               urcrnrlat=55,
    #               urcrnrlon=140,
    #               resolution='l', area_thresh=10000)

    # map.readshapefile('./data/gadm36_CHN_shp/gadm36_CHN_1', 'states', drawbounds=True)
    # map.readshapefile('./data/gadm36_HKG_shp/gadm36_HKG_1', 'states', drawbounds=True)
    # map.readshapefile('./data/gadm36_MAC_shp/gadm36_MAC_1', 'states', drawbounds=True)
    # map.readshapefile('./data/gadm36_TWN_shp/gadm36_TWN_1', 'states', drawbounds=True)
    fig = plt.figure()
    ax1 = fig.add_axes([0.1,0.1,0.8,0.8])#[left,bottom,width,height]
    map = Basemap(projection='mill',lat_0=36,lon_0=122,llcrnrlat=30.5 ,urcrnrlat=35.3,llcrnrlon=116.2,urcrnrlon=121.99,ax=ax1,rsphere=6371200.,resolution='h',area_thresh=1000000)
    shp_info = map.readshapefile('./data/gadm36_CHN_shp/gadm36_CHN_3','states',drawbounds=False)
    for info, shp in zip(map.states_info, map.states):
        #print('info',info)
       # print('shp',shp)
        proid = info['NAME_1']
        if proid == 'Jiangsu':
            poly = Polygon(shp,facecolor='w',edgecolor='k', lw=1.0, alpha=0.1)#注意设置透明度alpha，否则点会被地图覆盖
            ax1.add_patch(poly)
    posi = pd.read_csv('./data/city_text2.csv')
 #   lat = np.array(posi["纬度"][0:2362])#获取经纬度坐标，一共有48个数据
 #   lon = np.array(posi["经度"][0:2362])
    lat = np.array(posi["纬度"][0:74])#获取经纬度坐标，一共有48个数据
    lon = np.array(posi["经度"][0:74])
    # val = np.array(posi["val"][0:48],dtype=float) #获取数值
    #  size = (val-np.min(val)+0.05)*800#对点的数值作离散化，使得大小的显示明显
    x,y = map(lon,lat)
    map.scatter(x, y, s=1, color='green')  # 要标记的点的坐标、大小及颜色
    # for i in range(0,47):
    # plt.text(x[i]+5000,y[i]+5000,str(val[i]))
    # plt.text(lat[i],lon[i],str(val[i]), family='serif', style='italic', ha='right', wrap=True)
    # plt.annotate(s=3.33,xy=(x,y),xytext=None, xycoords='data',textcoords='offset points', arrowprops=None,fontsize=16)
    map.drawmapboundary(fill_color='aqua')  # 边界线
    # map.fillcontinents()
    map.drawstates()
    # map.drawcoastlines()  #海岸线
    # map.drawcountries()
    # map.drawcounties()
    plt.title('Yunnan')  # 标题

   # radius = [30,2,4,79,80]
    # 调用画圆函数
    for i in range(len(radius)):
        print('i',i)
        DrawPointMap(map,centerlon[i], centerlat[i], radius[i], lw=2.)

    # plt.savefig('Jiangsu.png', dpi=100, bbox_inches='tight')#文件命名为Jiangsu.png存储
    #plt.show()
'''

