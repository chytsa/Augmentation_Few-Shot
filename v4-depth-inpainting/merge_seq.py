import os


dirs = sorted(os.listdir('.'))
for dir in dirs:
    if dir == 'seq-02':
        img_fnames = sorted(os.listdir(dir))
        for img_fname in img_fnames:
            #filename, file_extension = os.path.splitext(img_fname)
            new_fname = '1' + "{:0>3d}".format(int(img_fname[:-4])) + '.png'
            #print(os.path.join(dir, new_fname))
            os.rename(os.path.join(dir, img_fname), os.path.join(dir, new_fname))
    if dir == 'seq-05':
        img_fnames = sorted(os.listdir(dir))
        for img_fname in img_fnames:
            #filename, file_extension = os.path.splitext(img_fname)
            new_fname = '2' + "{:0>3d}".format(int(img_fname[:-4])) + '.png'
            #print(os.path.join(dir, new_fname))
            os.rename(os.path.join(dir, img_fname), os.path.join(dir, new_fname))
    if dir == 'seq-07':
        img_fnames = sorted(os.listdir(dir))
        for img_fname in img_fnames:
            #filename, file_extension = os.path.splitext(img_fname)
            new_fname = '3' + "{:0>3d}".format(int(img_fname[:-4])) + '.png'
            #print(os.path.join(dir, new_fname))
            os.rename(os.path.join(dir, img_fname), os.path.join(dir, new_fname))
    if dir == 'seq-08':
        img_fnames = sorted(os.listdir(dir))
        for img_fname in img_fnames:
            #filename, file_extension = os.path.splitext(img_fname)
            new_fname = '4' + "{:0>3d}".format(int(img_fname[:-4])) + '.png'
            #print(os.path.join(dir, new_fname))
            os.rename(os.path.join(dir, img_fname), os.path.join(dir, new_fname))
    if dir == 'seq-11':
        img_fnames = sorted(os.listdir(dir))
        for img_fname in img_fnames:
            #filename, file_extension = os.path.splitext(img_fname)
            new_fname = '5' + "{:0>3d}".format(int(img_fname[:-4])) + '.png'
            #print(os.path.join(dir, new_fname))
            os.rename(os.path.join(dir, img_fname), os.path.join(dir, new_fname))
    if dir == 'seq-13':
        img_fnames = sorted(os.listdir(dir))
        for img_fname in img_fnames:
            #filename, file_extension = os.path.splitext(img_fname)
            new_fname = '6' + "{:0>3d}".format(int(img_fname[:-4])) + '.png'
            #print(os.path.join(dir, new_fname))
            os.rename(os.path.join(dir, img_fname), os.path.join(dir, new_fname))