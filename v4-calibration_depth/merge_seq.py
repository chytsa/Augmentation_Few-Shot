import os

scene = "office"

scene_list = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]

seq_list = [['seq-01', 'seq-02', 'seq-04', 'seq-06'],
        ['seq-01', 'seq-02'],
        ['seq-02'],
        ['seq-01', 'seq-03', 'seq-04', 'seq-05', 'seq-08', 'seq-10'],
        ['seq-02', 'seq-03', 'seq-06', 'seq-08'],
        ['seq-01', 'seq-02', 'seq-05', 'seq-07', 'seq-08', 'seq-11', 'seq-13'],
        ['seq-02', 'seq-03', 'seq-05', 'seq-06']]

multiple = 1000
if scene == "stairs":
    multiple = 500

for seq_id in range(len(seq_list[scene_list.index(scene)])):
    dir = seq_list[scene_list.index(scene)][seq_id]
    img_fnames = sorted(os.listdir(dir))
    for img_fname in img_fnames:
        #filename, file_extension = os.path.splitext(img_fname)
        new_fname = str(int(img_fname[:-4]) + seq_id * multiple) + '.png'
        #print(os.path.join(dir, new_fname))
        os.rename(os.path.join(dir, img_fname), os.path.join(dir, new_fname))