num = input("num of original dataset: ")
aug = input("aug: ")
scene = input("scene: ")
print(num, aug, scene)

f = open(scene+".txt", 'w')

for i in range(int(num)):
    for j in range(int(aug)):
        print(str(i)+"_"+str(j), file=f)

f.close()