from result1 import points_length
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pi = np.pi


def get_rect(a, b):
    a = np.array(a)
    b = np.array(b)
    u = get_normal(b - a)
    v = np.array([-u[1], u[0]])
    x = 0.275 * u
    y = 0.15 * v
    rect = [b + x + y, a - x + y, a - x - y, b + x - y]
    return rect


def get_normal(v):
    len = np.sqrt(np.dot(v, v))
    return v / len if len != 0 else v


def sat(body_a, body_b):
    return not find_separate_axis(body_a, body_a, body_b) and not find_separate_axis(body_b, body_a, body_b)


def find_separate_axis(lines_body, body_a, body_b):
    for i in range(np.shape(lines_body)[0] - 1):
        p1 = lines_body[i]
        p2 = lines_body[i] + get_normal(lines_body[i + 1] - lines_body[i])

        a_min, a_max = body_cast(p1, p2, body_a)

        b_min, b_max = body_cast(p1, p2, body_b)

        if p1[0] != p2[0]:
            max_min = a_min if a_min[0] > b_min[0] else b_min
            min_max = a_max if a_max[0] < b_max[0] else b_max
            if max_min[0] < min_max[0]:
                continue
        else:
            max_min = a_min if a_min[1] > b_min[1] else b_min
            min_max = a_max if a_max[1] < b_max[1] else b_max
            if max_min[1] < min_max[1]:
                continue

        return True

    return False


def body_cast(a, b, body):
    body_min = body_max = point_cast(a, b, body[0])
    for p in body:
        p_cast = point_cast(a, b, p)
        if p_cast[0] < body_min[0] or (p_cast[0] == body_min[0] and p_cast[1] < body_min[1]):
            body_min = p_cast
        if p_cast[0] > body_max[0] or (p_cast[0] == body_max[0] and p_cast[1] > body_max[1]):
            body_max = p_cast
    return body_min, body_max


def point_cast(a, b, p):
    u = p - a
    v = b - a
    p_cast = a + v * (np.dot(v, u) / np.dot(v, v))
    return p_cast


def overlap(rect):
    for i in range(2, np.shape(rect)[0]):
        if sat(rect[0], rect[i]):
            return i
    return -1


df = pd.read_excel('output_loc_1.xlsx')
data = df.to_numpy()
points = []
for j in range(points_length):
    list = []
    for i in range(224):
        list.append([data[i * 2][j], data[i * 2 + 1][j]])
    points.append(list)
rects = []
for stp in range(points_length):
    rect_set = []
    for j in range(223):
        rect_set.append(get_rect(points[stp][j], points[stp][j + 1]))
    rects.append(rect_set)
    flag = overlap(rect_set)
    if flag != -1:
        print(stp, flag, "重合")
    else:
        print(stp, "不重合")
