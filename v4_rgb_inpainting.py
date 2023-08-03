import numpy as np
import os
import cv2

np.random.seed(777)

data_dir = 'data'
dataset = '7Scenes'
scene = 'fire'

augtime = "64"

few_shot = "0.5"

aug = 'train_aug_'+few_shot+'-'+augtime+'-k4_cali-set-point-size-2.0'

few_shot_postfix = '_aug_'+augtime+'_few_shoot_'+few_shot

depth_postfix = ''
#depth_postfix = 'd_'

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
f = open(os.path.join(aug_dir, scene + few_shot_postfix + '.txt'))
aug_dir = os.path.join(aug_dir, aug)

output_dir = './result'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
    
for fname in f:
    fname = fname.splitlines()[0]
    print(fname)
    print(aug_dir, fname + '.mask.png')
    mask_fn = os.path.join(aug_dir, fname + '.mask.png')
    mask = cv2.imread(mask_fn, -1)
    mask = mask ^ 1
    mask = mask * 255
    mask = mask.astype(np.uint8)
    """
    mask_before_fn = os.path.join(output_dir, fname + '.mask_before.png')
    cv2.imwrite(mask_before_fn, mask)
    """
    """
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i][j] == 255:
                mask[i][j] = 0
            else:
                break
        for j in range(mask.shape[1]-1, -1, -1):
            if mask[i][j] == 255:
                mask[i][j] = 0
            else:
                break
    mask_after_fn = os.path.join(output_dir, fname + fname + '.mask_after.png')
    cv2.imwrite(mask_after_fn, mask)
    """
    img_fn = os.path.join(aug_dir, fname + '.color.png')
    img = cv2.imread(img_fn)
    
    #print(img.dtype)
    
    # cv2.INPAINT_TELEA
    # cv2.INPAINT_NS
    
    """
    RADIOUS = 3
    
    inpainting = cv2.inpaint(img, mask, RADIOUS, cv2.INPAINT_TELEA)
    img_inpaint_fn = os.path.join(output_dir, fname + '.color_'+depth_postfix+'t3.png')
    cv2.imwrite(img_inpaint_fn, inpainting)
    
    inpainting = cv2.inpaint(img, mask, RADIOUS, cv2.INPAINT_NS)
    img_inpaint_fn = os.path.join(output_dir, fname + '.color_'+depth_postfix+'n3.png')
    cv2.imwrite(img_inpaint_fn, inpainting)
    """
    RADIOUS = 4
    """
    inpainting = cv2.inpaint(img, mask, RADIOUS, cv2.INPAINT_TELEA)
    img_inpaint_fn = os.path.join(output_dir, fname + '.color_'+depth_postfix+'t4.png')
    cv2.imwrite(img_inpaint_fn, inpainting)
    """
    inpainting = cv2.inpaint(img, mask, RADIOUS, cv2.INPAINT_NS)
    img_inpaint_fn = os.path.join(output_dir, fname + '.color_'+depth_postfix+'n4.png')
    cv2.imwrite(img_inpaint_fn, inpainting)
    

    #img_v_fn = os.path.join(output_dir, fname + fname + '.color.png')
    #cv2.imwrite(img_v_fn, img)
    
f.close()
