num = input("num of original dataset: ")
aug = input("aug times: ")
print(num, aug)

few_shoot = [100, 50, 25, 10, 5, 4, 2, 1, 0.5]

for ff in few_shoot:
    f = open("aug_"+str(aug)+"_few_shoot_"+str(ff)+".txt", 'w')

    for i in range(0, int(num), int(100/ff)):
        for j in range(int(aug)):
            print(str(i)+"_"+str(j), file=f)

    f.close()

