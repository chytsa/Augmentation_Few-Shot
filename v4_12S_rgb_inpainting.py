import numpy as np
import os
import cv2

data_dir = 'data'
dataset = '12Scenes'
scene1 = 'apt1'
scene2 = 'kitchen'
aug = 'train_aug_1-30-set-point-size-2.0'


list = 'aug_32_few_shoot_1.txt'

aug_dir = os.path.join(data_dir, dataset)
aug_dir = os.path.join(aug_dir, scene1)
aug_dir = os.path.join(aug_dir, scene2)
aug_dir2 = os.path.join(aug_dir, aug)

txt = os.path.join(aug_dir, list)

f = open(txt)

output_dir = aug_dir2
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for fname in f:
    fname = fname.splitlines()[0]
    mask_fn = os.path.join(aug_dir2, fname + '.mask.png')
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
    img_fn = os.path.join(aug_dir2, fname + '.color.png')
    img = cv2.imread(img_fn)
    
    #print(img.dtype)
    
    # cv2.INPAINT_TELEA
    # cv2.INPAINT_NS
    """
    RADIOUS = 2
    inpainting = cv2.inpaint(img, mask, RADIOUS, cv2.INPAINT_TELEA)
    img_inpaint_fn = os.path.join(output_dir, fname + '.color_t2.png')
    cv2.imwrite(img_inpaint_fn, inpainting)
    
    inpainting = cv2.inpaint(img, mask, RADIOUS, cv2.INPAINT_NS)
    img_inpaint_fn = os.path.join(output_dir, fname + '.color_n2.png')
    cv2.imwrite(img_inpaint_fn, inpainting)
    """
    RADIOUS = 3
    
    inpainting = cv2.inpaint(img, mask, RADIOUS, cv2.INPAINT_TELEA)
    img_inpaint_fn = os.path.join(output_dir, fname + '.color_t3.png')
    cv2.imwrite(img_inpaint_fn, inpainting)
    
    inpainting = cv2.inpaint(img, mask, RADIOUS, cv2.INPAINT_NS)
    img_inpaint_fn = os.path.join(output_dir, fname + '.color_n3.png')
    cv2.imwrite(img_inpaint_fn, inpainting)
    
    RADIOUS = 4
    inpainting = cv2.inpaint(img, mask, RADIOUS, cv2.INPAINT_TELEA)
    img_inpaint_fn = os.path.join(output_dir, fname + '.color_t4.png')
    cv2.imwrite(img_inpaint_fn, inpainting)
    
    inpainting = cv2.inpaint(img, mask, RADIOUS, cv2.INPAINT_NS)
    img_inpaint_fn = os.path.join(output_dir, fname + '.color_n4.png')
    cv2.imwrite(img_inpaint_fn, inpainting)
    """
    RADIOUS = 5
    inpainting = cv2.inpaint(img, mask, RADIOUS, cv2.INPAINT_TELEA)
    img_inpaint_fn = os.path.join(output_dir, fname + '.color_t5.png')
    cv2.imwrite(img_inpaint_fn, inpainting)
    
    inpainting = cv2.inpaint(img, mask, RADIOUS, cv2.INPAINT_NS)
    img_inpaint_fn = os.path.join(output_dir, fname + '.color_n5.png')
    cv2.imwrite(img_inpaint_fn, inpainting)
    """
    

    #img_v_fn = os.path.join(output_dir, fname + fname + '.color.png')
    #cv2.imwrite(img_v_fn, img)
    
f.close()
