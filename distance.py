# Calculate the distance between two latitudes and longitudes

import math
from numpy import mat

# 3.6378.137 is the radius of the Earth in kilometers;
EARTH_REDIUS = 6378.137

# Define the calculation method of the circumference
def rad(d):
    #print('d',d)
    #print(type(d))
    rad = float(d) * math.pi / 180.0
    #print(rad)
    return rad

# Define the distance between two latitudes and longitudes (using the Google Map Distance calculation method)
def getDistance(lat1, lng1, lat2, lng2):      #  Lat1 Lung1 represents point A latitude and longitude, Lat2 Lung2 represents point B latitude and longitude
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    #print(a)
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(math.sin(a/2), 2) + math.cos(radLat1) * math.cos(radLat2) * math.pow(math.sin(b/2), 2)))
    s = s * EARTH_REDIUS
    return s

#Distance matrix L generation function
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
