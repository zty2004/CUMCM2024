from mpmath import mp
import pandas as pd

mp.dps = 50
p = 0.55
alpha = p / (2 * mp.pi)
theta0 = 32 * mp.pi
r0 = 16 * p
dragon_head_length = (341 - 27.5 * 2) / 100
dragon_length = (220 - 27.5 * 2) / 100

# def fakecos(x):
#    return 1 - mp.power(x, 2) / 2 + mp.power(x, 4) / 24 - mp.power(x, 6) / 720


def polar2cartesian(theta):
    return [alpha * theta * mp.cos(theta), alpha * theta * mp.sin(theta)]


def length(theta):
    tmp = mp.sqrt(1 + mp.power(theta, 2))
    return alpha * (0.5 * mp.ln(theta + tmp) + 0.5 * theta * tmp)


def find_dragon_head_1(t):
    return mp.findroot(lambda x: length(theta0) - length(x) - t, 0)


def find_dragon_next(theta, dist):
    delta_theta = mp.findroot(lambda x: ((2 * mp.power(theta, 2) + 2 * theta * x) * (
        1 - mp.cos(x)) + mp.power(x, 2)) - mp.power(dist/alpha, 2), 0.1)
    return theta + delta_theta

# def find_dragon_next_2(theta, dist):
#    delta_theta = mp.findroot(lambda x: mp.power(theta, 2) + mp.power(theta + x, 2) - 2 * theta * (theta + x) * fakecos(x) - mp.power(dist/alpha, 2), 0.3)
#    return theta + delta_theta


def find_dragon_theta(t):
    dragon = find_dragon_head_1(t)
    ans = [dragon]
    dragon = find_dragon_next(dragon, dragon_head_length)
    for i in range(1, 224):
        ans.append(dragon)
        dragon = find_dragon_next(dragon, dragon_length)
    return ans


def find_dragon(t):
    ans = []
    dragon = find_dragon_theta(t)
    for i in dragon:
        ans.append(polar2cartesian(i))
    return ans


def find_dragon_velocity(t):
    deltat = 1e-6
    dragon1 = find_dragon(t)
    dragon2 = find_dragon(t + deltat)
    v = []
    for i in range(len(dragon1)):
        vx = mp.fdiv(mp.fsub(dragon2[i][0], dragon1[i][0]), deltat)
        vy = mp.fdiv(mp.fsub(dragon2[i][1], dragon1[i][1]), deltat)
        vv = mp.sqrt(vx**2 + vy**2)
        v.append(vv)
    return v


def dLdtheta(x):
    return (2 * x**3 + mp.sqrt(1 + x**2) * (2 * x**2 + 1) + 2 * x) / (mp.sqrt(1 + x**2) * 2 * x + 2 * x**2 + 1)


def find_dragon_fake_velocity(t):
    dragon = find_dragon_theta(t)
    v = []
    for i in dragon:
        v.append(mp.fdiv(dLdtheta(dragon[0]), dLdtheta(i)))
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
