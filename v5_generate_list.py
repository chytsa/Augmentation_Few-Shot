scene = input("scene: ")
num = input("num of original dataset: ")
aug = input("aug times: ")
print(scene, num, aug)

few_shoot = [50, 25, 10, 5, 4, 2, 1, 0.5]

for ff in few_shoot:
    f = open(scene+"_aug_"+str(aug)+"_few_shoot_"+str(ff)+".txt", 'w')

    for i in range(0, int(num), int(100/ff)):
        for j in range(int(aug)):
            print(str(i)+"_"+str(j), file=f)

    f.close()

