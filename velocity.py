from setup import *

velocity = []
for i in range(301):
    velocity.append(find_dragon_velocity(i))

fake_velocity = []
for i in range(301):
    fake_velocity.append(find_dragon_fake_velocity(i))

'''
for i in range(301):
    dv = [a - b for a, b in zip(velocity[i], fake_velocity[i])]
print(dv)
'''

datav = {}
for i in range(301):
    key = str(i) + 's'
    list = []
    for j in velocity[i]:
        list.append(format(float(mp.nstr(j, 8)), '.6f'))
    datav[key] = list

# print(datav)
dfv = pd.DataFrame(datav)
dfv.to_excel('output_v_1.xlsx', index=False)
