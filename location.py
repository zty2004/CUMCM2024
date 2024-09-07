from setup import *

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
df.to_excel('output_loc.xlsx', index=False)
