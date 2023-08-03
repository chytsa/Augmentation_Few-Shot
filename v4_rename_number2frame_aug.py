import numpy as np
import os

data_dir = 'data'
dataset = '7Scenes'
scene = 'fire'
aug = 'train_aug_0.5-64-k4_cali-set-point-size-2.0'

"""
[data]:
    [7Scenes]:
        sensorTrans.txt
        [heads]:
            centers.npy
            translation.txt
            heads.txt(get by ls > heads.txt, and replace '.png' by empty)
            [train_aug]:
                color
                depth
                pose
"""

# 1: adjust train_aug
# 1-1: add color/depth postfix, and put them together

aug_dir = os.path.join(data_dir, dataset)
aug_dir = os.path.join(aug_dir, scene)
aug_dir = os.path.join(aug_dir, aug)


color_dir = os.path.join(aug_dir, 'color')
color_imgs = os.listdir(color_dir)
for color_img in color_imgs:
    filename, file_extension = os.path.splitext(color_img)
    new_fname = filename + '.color.png'

    os.rename(os.path.join(color_dir, color_img), os.path.join(aug_dir, new_fname))

depth_dir = os.path.join(aug_dir, 'depth')
depth_imgs = os.listdir(depth_dir)
for depth_img in depth_imgs:
    filename, file_extension = os.path.splitext(depth_img)
    new_fname = filename + '.depth.png'

    os.rename(os.path.join(depth_dir, depth_img), os.path.join(aug_dir, new_fname))

mask_dir = os.path.join(aug_dir, 'mask')
mask_imgs = os.listdir(mask_dir)
for mask_img in mask_imgs:
    filename, file_extension = os.path.splitext(mask_img)
    new_fname = filename + '.mask.png'

    os.rename(os.path.join(mask_dir, mask_img), os.path.join(aug_dir, new_fname))

pose_dir = os.path.join(aug_dir, 'pose')
pose_imgs = os.listdir(pose_dir)
for pose_img in pose_imgs:
    filename, file_extension = os.path.splitext(pose_img)
    new_fname = filename + '.pose.txt'

    os.rename(os.path.join(pose_dir, pose_img), os.path.join(aug_dir, new_fname))

rotate_dir = os.path.join(aug_dir, 'rotate')
rotate_imgs = os.listdir(rotate_dir)
for rotate_img in rotate_imgs:
    filename, file_extension = os.path.splitext(rotate_img)
    new_fname = filename + '.rotate.txt'

    os.rename(os.path.join(rotate_dir, rotate_img), os.path.join(aug_dir, new_fname))
    
os.rmdir(color_dir)
os.rmdir(depth_dir)
os.rmdir(mask_dir)
os.rmdir(pose_dir)
os.rmdir(rotate_dir)
