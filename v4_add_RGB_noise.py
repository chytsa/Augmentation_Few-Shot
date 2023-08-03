import numpy as np
import os
import cv2

np.random.seed(777)

data_dir = 'data'
dataset = '7Scenes'
scene = 'stairs'
aug = 'train_aug_1-64-k4_cali-set-point-size-2.0'
scene_postfix = '_aug_64_few_shoot_1.txt'

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

aug_dir = os.path.join(data_dir, dataset)
aug_dir = os.path.join(aug_dir, scene)
f = open(os.path.join(aug_dir, scene+scene_postfix))
aug_dir = os.path.join(aug_dir, aug)

for fname in f:
    fname = fname.splitlines()[0]
    rgb_name = os.path.join(aug_dir, fname+'.color.png')
    mask_name = os.path.join(aug_dir, fname+'.mask.png')
    
    img = cv2.imread(rgb_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    mask = cv2.imread(mask_name, -1)
    noise_mask = np.zeros_like(mask)
    noise_mask[mask == 0] = 1.0
    """
    new_mask = noise_mask.astype(np.uint8)
    fname = '0.mask.png'
    cv2.imwrite(fname, new_mask)
    """
    noise_mask = np.repeat(noise_mask[:, :, np.newaxis], 3, axis=2)
    
    rgb_mask = np.ones_like(mask)
    rgb_mask[mask == 0] = 0.0
    rgb_mask = np.repeat(rgb_mask[:, :, np.newaxis], 3, axis=2)
    
    noise = np.random.rand(img.shape[0], img.shape[1], img.shape[2])
    noise *= 255
    noise = noise * noise_mask
    #fname = '0.noise.png'
    #cv2.imwrite(fname, noise)
    
    img = img * rgb_mask
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    #print(img.shape)
    #print(noise.shape)
    
    img = img + noise
    
    #print(img.shape)
    
    fname = os.path.join(aug_dir, fname+'.color_noise.png')
    cv2.imwrite(fname, img)
