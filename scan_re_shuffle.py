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

pd.set_option('display.width', 1000)  # Set the character display width
pd.set_option('display.max_rows', None)  # Set display maximum line
pd.set_option('display.max_columns', None)  # Set display maximum line


def space_cluster(df1, R_max):
    d = []  # Distance calculation

    # Get the latitude and longitude that meets the conditions
    def JD(k):  # Define longitude
        # print(df1.序号)
        a = df1[df1.序号 == k].经度
        return a.tolist()[0]

    def WD(k):  # 定义纬度
        b = df1[df1.序号 == k].纬度
        # print(b)
        return b.tolist()[0]

    m, n = df1.shape
    print('m', 'n', m, n)
    # Computing space aggregation area
    for i in range(m):
        d.append([])
        for j in range(m):
            # print(JD(i+1),WD(i+1),JD(j+1),WD(j+1))
            d[i].append(distance.getDistance(JD(i + 1), WD(i + 1), JD(j + 1), WD(j + 1)))  # 计算当前地点与别的地址的距离
        # print('i',i,','"d[i]",d )
        # print(d)

        for k in range(m):
            # print(d[i][k])
            if (d[i][k] <= R_max):  # When the distance between the two locations is less than the maximum radius, draw the circle with i as the center and k as the radius.
                c = {}
                c["center_xbid"] = i + 1
                c["r"] = d[i][k]
                xbid = []  # 数组存储下标

                # print(d[i][k])
                for v in range(m):  # When i is the center of the circle and k is the radius, draw a circle, and judge the location within the circle to store the subscript set.
                    # print(d[i][v])
                    if (d[i][k] >= d[i][v]):
                        if (i != k):
                            xbid.append(v + 1)  # Spatial clustering set
                        else:
                            xbid.append(i + 1)  # Store only the center of the circle
                    else:
                        continue

                #if i == 170:
                   # print('xbid-171', xbid)
                c["xbids"] = xbid
                circle.append(c)  # Storage form [{ }{ }{ }]
            else:
                continue
    # print(circle)
    print('d', d)
    return circle


# print(circle)
# print(circle)
def scan(case, circle, zdsjc, U, topN, day):  # Scan result calculation function

    C = copy.deepcopy(case)  # Event matrix
    circle = copy.deepcopy(circle)  # Center radius and space cluster record list
    T_max = copy.deepcopy(zdsjc)  # Maximum time cluster
    U = copy.deepcopy(U)  # Expectation matrix
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
            circle1[i]["T_cluster"] = T_cluster + 1  # Time cluster
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
                LGLR = 0  # The actual number of events is less than expected, normal warning
            else:
                LGLR = C_zd * math.log(C_zd / u_zd) + (C_sum - C_zd) * math.log(
                    (C_sum - C_zd) / (C_sum - u_zd))  # Calculating the likelihood ratio
            circle1[i]["LGLR"] = LGLR  # circle1[i]Add likelihood ratio key
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
    # sorted_cir = sorted(cir, key=operator.itemgetter("LGLR"),reverse=True)   # Sort LGLR from big to small first

    ### Turn df --" -- 》 remove r = 0, LGLR = 0 --" Sort -- " Same value to weight --"
    df_cir = pd.DataFrame(cir)

    # First sort by LGLR, then sort by r
    df_cir = df_cir.sort_index(axis=0, ascending=[False, True], by=['LGLR', 'r'])

    ## Remove the weight and remove the points contained in the first gathering circle
    f2 = df_cir.iloc[[0]]
    xb = f2.center_xbid.tolist()

    list1 = f2.xbids.tolist()
    list = copy.deepcopy(list1)

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


    df_cir = df_cir.drop_duplicates(subset=['LGLR'], keep='first')  # After the likelihood ratio is sorted from large to small, the equal likelihood is sorted from its small to large radius, leaving the first one to delete the rest.
    # = df_cir.drop_duplicates(subset=['r'], keep='first')
    df_cir = df_cir.reset_index(drop=True)

    # Take 10 elements
    df_cir = df_cir.iloc[0:10]
    # print('df_cir',df_cir)
    # df_cir = df_cir.tolist()
    # df_cir = np.array(df_cir)  # np.ndarray()
    # df_cir = df_cir.tolist()  # list
    # print(df_cir)
    return df_cir

# Knuth-Durstenfeld Shuffle algorithm
def randpermBySYB(p1):
    p = copy.copy(p1)
    p = np.array(p)
    # print(type(p))
    # print(p.shape[0])
    n = np.size(p)
    # print(n)
    # if np.size(p,0)==1:    # Judgment is a row matrix
    for i in range(n):
        # print('i',i)
        j = np.random.randint(n - i)  # Generate pseudo-random integers in the range 1 to n-1
        # print('j',j)
        tmp = p[i + j]
        p[i + j] = p[i]
        p[i] = tmp  # Exchange data
    return p


## Random rearrangement
def shuffle(v):
    v = np.array(v)
    # print(v[0])
    m, n = v.shape
    # print('m,',m)
    for i in range(m):
        # print('v[i]',v[i])
        v[i] = randpermBySYB(v[i])
    return v

#Visual test module
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
    #  Address file address
    # data_dz = r"data\city_new.csv"
    #  Case number file address
    # data_case = r"data\fenbei_data_new.csv"

    data_dz = r"data\city_jjs.csv"
    data_case = r"data\jjs_text2.csv"

    # Read location file
    df1 = pd.DataFrame(pd.read_csv(data_dz))
    # Read case file
    df2 = pd.DataFrame(pd.read_csv(data_case))

    # print(df2.date)
    # print(xh[0])0
    print(df1)
    print(df2)

    # m Total number of reference addresses
    m, n = df1.shape
    # print(m）
    # initialization
    R_max = 6  # Maximum radius
    T_max = 3  # Maximum time cluster
    # day = 7    # Reference days
    topN = 10  # Take the top 10 GLR
    # xbid = []         # Storage space cluster subscript
    space_xbis = []  # Store all space clusters
    # c = {}           # The dictionary stores the circle data, and the center of the circle, the radius, and the space within the circle contain the address.
    circle = []  # Stores the center, radius, and space subscript aggregates for each scan window

    starttime = "2018/2/1 0:00:00"  # Starting time
    endtime = "2018/2/7 23:59:59"  # End Time

    # starttime = time.mktime(time.strptime(starttime, '%Y-%m-%d %H:%M:%d'))
    # endtime = time.mktime(time.strptime(endtime, '%Y-%m-%d %H:%M:%d'))
    starttime1 = time.mktime(time.strptime(starttime, '%Y/%m/%d %H:%M:%S'))  # time.mktime(tupletime)Accept time tuples and return timestamps
    endtime1 = time.mktime(time.strptime(endtime,'%Y/%m/%d %H:%M:%S'))  # time.strptime(str,fmt='%a %b %d %H:%M:%S %Y') Parse a time string into a time tuple according to the format of fmt.根据fmt的格式把一个时间字符串解析为时间元组。
    work_days = int((endtime1 - starttime1) / (24 * 60 * 60))  # Milliseconds to day
    day = work_days + 1  # Reference days
    print('day', day)
    # T_cluster = 0  # Time cluster is initially 0
    C, C_sum, U, C_z, C_d = Csum_ready.C_zd(starttime, endtime, df2, df1)  # Calculate C, C_sum, u
    # Distance space array after judgment
    circle = space_cluster(df1, R_max)
    print('C', C)
    # print('C_sum',C_sum)
    #  print('C_d',C_d)

    # print('U',U)
    print('circle', circle)

    result_init = scan(C, circle, T_max, U, topN, day)  # Raw data result
    print("result_init")
    print(result_init)

    result_init = np.array(result_init)
    result_init = result_init.tolist()
    # print('result_init',result_init)

    fzcs = 999  # Number of simulations
    c_re = np.zeros((1, day))  # initialization
    c_re = c_re.astype(np.int)
    np.random.seed(0)  # Randomly the same 

    # Store the final result
    zzjg = []
    # Take raw data, compare and get P value
    # for c in result_init:
    count = 0
    fz = []
    # print('c[1]', c[1])
    c = result_init[0]
    for s in range(fzcs):
        c_re = shuffle(C)
        ## Calculate the expected value
        c_re = np.array(c_re)  # List to array
        c_re_sum = c_re.sum()  # C Summation
        c_re_z = c_re.sum(axis=1)  # C_z specifies the total number of events on the area z, 1 indicates the line
        c_re_d = c_re.sum(axis=0)  # C_d specifies the number of all-segment events on the time period d, the number of events in the whole zone, and 0 indicates the column
        u_re_zd = np.multiply(c_re_d, np.mat(c_re_z).T) / c_re_sum  # Calculate the expected value
        u_re_zd = np.array(u_re_zd)  # List to array

        # Monte Carlo scan
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
    # Store the final result
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

    # Convert to dataFrame for easy viewing
    zzjg = pd.DataFrame(zzjg, columns=['C_zd', 'LGLR', 'T_cluster', 'center_xbid', 'r', 'u_zd', 'xbids', 'P-value'])
    print(zzjg)
    data_case = r"temp\smjg_0201_0207_7_10.xlsx"
    zzjg.to_excel(data_case)

    print(zzjg)

    zzjg = zzjg.iloc[0:5]

