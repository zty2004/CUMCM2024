import mpmath as mp
import numpy as np
pi = mp.pi
dragon_head_length = (341 - 27.5 * 2) / 100
dragon_length = (220 - 27.5 * 2) / 100


def collision(p, theta0):
    alpha = p / (2 * pi)

    def polar2cartesian(theta):
        return [alpha * theta * mp.cos(theta), alpha * theta * mp.sin(theta)]

    def convert_dragon(list):
        ans = []
        for i in list:
            list1 = []
            for j in i:
                list1.append(float(format(float(mp.nstr(j, 8)), '.6f')))
            ans.append(list1)
        return ans

    def find_dragon_next(theta, dist):
        delta_theta = mp.findroot(lambda x: ((2 * mp.power(theta, 2) + 2 * theta * x) * (
            1 - mp.cos(x)) + mp.power(x, 2)) - mp.power(dist/alpha, 2), 0.1)
        return theta + delta_theta

    def find_dragon(theta):
        dragon = theta
        ans = [polar2cartesian(dragon)]
        dragon = find_dragon_next(dragon, dragon_head_length)
        for i in range(1, 224):
            ans.append(polar2cartesian(dragon))
            dragon = find_dragon_next(dragon, dragon_length)
        return ans

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

    dragon = convert_dragon(find_dragon(theta0))
    rect = []
    for i in range(0, len(dragon) - 1):
        rect.append(get_rect(dragon[i], dragon[i + 1]))
    if overlap(rect) == -1:
        return False
    else:
        return True


def linear_find(p):
    for i in np.arange(67, 40, -0.1):
        if collision(p, i):
            return i
    return 32 * pi


def lower_bound(p):
    l = 0
    r = 32 * pi
    eps = 1e-6
    while r - l > eps:
        m = (l + r) / 2
        if collision(p, m):
            l = m
        else:
            r = m
    return l


r0 = 4.5
lp = 0.442
rp = 0.443
eps = 1e-6
while (rp - lp > eps):
    m = (lp + rp) / 2
    theta0 = 2 * pi * r0 / m
    print(m, lower_bound(m), theta0)
    if lower_bound(m) - theta0 > eps:
        lp = m
    else:
        rp = m
print(lp)
