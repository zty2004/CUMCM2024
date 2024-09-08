import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import mpmath as mp
pi = np.pi
p = 1.7
r0 = 4.5
k = 2
alpha = p / (2 * pi)
dragon_head_length = (341 - 27.5 * 2) / 100
dragon_length = (220 - 27.5 * 2) / 100


def collision(t):
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

    def find_dragon_head_1(t):

    def find_dragon_next(theta, dist):
        delta_theta = mp.findroot(lambda x: ((2 * mp.power(theta, 2) + 2 * theta * x) * (
            1 - mp.cos(x)) + mp.power(x, 2)) - mp.power(dist/alpha, 2), 1 / theta)
        return theta + delta_theta

    def find_dragon(t):
        dragon = find_dragon_head_1(t)
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

    dragon = convert_dragon(find_dragon(t))
    rect = []
    for i in range(0, len(dragon) - 1):
        rect.append(get_rect(dragon[i], dragon[i + 1]))
    if overlap(rect) == -1:
        return False
    else:
        return True


def norm(theta, flag):
    c = -flag / np.sqrt(1 + theta**2)
    return c * np.array([np.sin(theta) + theta * np.cos(theta), -np.cos(theta) + theta * np.sin(theta)])


def get_center(x, y, radius):
    nx = norm(x, 1)
    ny = norm(y, -1)
    xx = alpha * np.array([x * np.cos(x), x * np.sin(x)])
    yy = -alpha * np.array([y * np.cos(y), y * np.sin(y)])
    cx = xx + k * radius * nx
    cy = yy + radius * ny
    return xx, yy, cx, cy


def get_length(x, y, radius):
    _, _, cx, cy = get_center(x, y, radius)
    return np.sqrt(np.dot(cy - cx, cy - cx))


def get_radius(x, y):
    l = 0
    r = 4.5
    eps = 1e-6
    while (r - l > eps):
        m = (l + r) / 2
        if get_length(x, y, m) - (k + 1) * m > eps:
            l = m
        else:
            r = m
    return l


def get_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cos_theta = dot_product / (norm_v1 * norm_v2)
    angle_radians = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return angle_radians


def s(x, y):
    v1 = np.array([np.cos(x), np.sin(x)])
    v2 = -np.array([np.cos(y), np.sin(y)])
    if get_angle(v1, v2) < pi / 2 or x[i] < y[j]:
        return -1
    r = get_radius(x, y)
    xx, yy, cx, cy = get_center(x, y, r)
    cut = 1 / (1 + k) * cx + k / (1 + k) * cy
    a = xx - cx
    b = cut - cx
    c = cut - cy
    d = yy - cy
    theta1 = get_angle(a, b)
    theta2 = get_angle(c, d)
    if np.cross(a, b) > 0:
        theta1 = 2 * pi - theta1
    if np.cross(c, d) < 0:
        theta2 = 2 * pi - theta2

    return theta1 * r * k + theta2 * r


theta0 = 2 * r0 * pi / p
sz = 200
mn = 100
ans = []
x = np.linspace(6.873472, theta0, sz)
y = np.linspace(6.873472, theta0, sz)
X, Y = np.meshgrid(x, y)
Z = np.zeros([sz, sz])
for i in range(sz):
    for j in range(sz):
        Z[i][j] = s(x[i], y[j])
        if Z[i][j] != -1 and Z[i][j] < mn:
            mn = Z[i][j]
            ans = [x[i], y[j], mn]
        print(x[i], y[j], Z[i][j])
print(ans)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

# 设置标题和标签
ax.set_title("3D Surface Plot")
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

plt.show()
