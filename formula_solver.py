from mpmath import mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

mp.dps = 50
alpha = 0.55 / (2 * mp.pi)
theta0 = 32 * mp.pi
r0 = 16 * 0.55
dragon_head_length = (341 - 27.5 * 2) /100
dragon_length = (220 - 27.5 * 2) / 100

#def fakecos(x):
#    return 1 - mp.power(x, 2) / 2 + mp.power(x, 4) / 24 - mp.power(x, 6) / 720

def polar2cartesian(theta):
    return [alpha * theta * mp.cos(theta), alpha * theta * mp.sin(theta)]

def length(theta):
    tmp = mp.sqrt(1 + mp.power(theta, 2))
    return alpha * (0.5 * mp.ln(theta + tmp) + 0.5 * theta * tmp)

def find_dragon_head_1(t):
    return mp.findroot(lambda x: length(theta0) - length(x) - t, 0)

def find_dragon_next(theta, dist):
    delta_theta = mp.findroot(lambda x: ((2 * mp.power(theta, 2) + 2 * theta * x) * (1 - mp.cos(x)) + mp.power(x, 2)) - mp.power(dist/alpha, 2), 0.1)
    return theta + delta_theta

#def find_dragon_next_2(theta, dist):
#    delta_theta = mp.findroot(lambda x: mp.power(theta, 2) + mp.power(theta + x, 2) - 2 * theta * (theta + x) * fakecos(x) - mp.power(dist/alpha, 2), 0.3)
#    return theta + delta_theta

def find_dragon(t):
    dragon = find_dragon_head_1(t)
    ans = [polar2cartesian(dragon)]
    dragon = find_dragon_next(dragon, dragon_head_length)
    for i in range(1, 224):
        ans.append(polar2cartesian(dragon))
        dragon = find_dragon_next(dragon, dragon_length)
    return ans

def find_dragon_velocity(t):
    deltat = 1e-8
    dragon1 = find_dragon(t)
    dragon2 = find_dragon(t + deltat)
    v = []
    for i in range(len(dragon1)):
        vx = mp.fdiv(mp.fsub(dragon2[i][0], dragon1[i][0]), deltat);
        vy = mp.fdiv(mp.fsub(dragon2[i][1], dragon1[i][1]), deltat);
        vv = format(float(mp.nstr(mp.sqrt(vx**2 + vy**2), 8)), '.6f')
        v.append(vv)
    return v

def convert_dragon(list):
    ans = []
    for i in list:
        list1 = []
        for j in i:
            list2 = []
            for k in j:
                list2.append(format(float(mp.nstr(k, 8)), '.6f'))
            list1.append(list2)
        ans.append(list1)
    return ans

#location
points = []
for i in range(301):
    points.append(find_dragon(i))

points = convert_dragon(points)

data = {}
for i in range(301):
    key = str(i) + 's'
    list = []
    for j in points[i]:
        for k in j:
            list.append(k)
    data[key] = list

print(data)
df = pd.DataFrame(data)
df.to_excel('output_loc.xlsx', index = False)

#velocity
velocity = []
for i in range(301):
    velocity.append(find_dragon_velocity(i))

datav = {}
for i in range(301):
    key = str(i) + 's'
    list = []
    for j in velocity[i]:
        list.append(j);
    datav[key] = list
    
print(datav)
dfv = pd.DataFrame(datav)
dfv.to_excel('output_v.xlsx', index = False)