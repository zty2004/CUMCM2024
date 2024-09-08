
sz = 301

velocity = []
for i in range(sz):
    velocity.append(find_dragon_velocity(i))

fake_velocity = []
for i in range(sz):
    fake_velocity.append(find_dragon_fake_velocity(i))
print(fake_velocity)

dv = []
for i in range(sz):
    dv.append([a - b for a, b in zip(velocity[i], fake_velocity[i])])
# print(dv)


datav = {}
for i in range(sz):
    key = str(i) + 's'
    list = []
    for j in velocity[i]:
        list.append(format(float(mp.nstr(j, 8)), '.6f'))
    datav[key] = list

print(datav)
dfv = pd.DataFrame(datav)
dfv.to_excel('output_v_1.xlsx', index=False)
