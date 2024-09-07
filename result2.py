from setup import *
points = []
t = 412.473843
points.append(find_dragon(t))

points = convert_dragon(points)

data = {}
points_length = len(points[0])
for i in range(2):
    list = []
    key = str(i)
    for j in range(points_length):
        list.append(points[0][j][i])
        data[key] = list

print(data)
df = pd.DataFrame(data)
df.to_excel('output_loc_2.xlsx', index=False)

velocity = []
velocity.append(find_dragon_velocity(t))

datav = {}
list = []
for j in velocity[0]:
    list.append(format(float(mp.nstr(j, 8)), '.6f'))
datav['3'] = list

# print(datav)
dfv = pd.DataFrame(datav)
dfv.to_excel('output_v_2.xlsx', index=False)
