from __future__ import division

import torch
import numpy as np
import random
from imgaug import augmenters as iaa

def get_depth(depth, calibration_extrinsics, intrinsics_color,
              intrinsics_depth_inv):
    """Return the calibrated depth image (7-Scenes). 
    Calibration parameters from DSAC (https://github.com/cvlab-dresden/DSAC) 
    are used.
    """
    img_height, img_width = depth.shape[0], depth.shape[1]
    depth_ = np.zeros_like(depth)
    depthGG = depth
    x = np.linspace(0, img_width-1, img_width)
    y = np.linspace(0, img_height-1, img_height)
    xx, yy = np.meshgrid(x, y)
    xx = np.reshape(xx, (1, -1))
    yy = np.reshape(yy, (1, -1))
    ones = np.ones_like(xx)
    pcoord_depth = np.concatenate((xx, yy, ones), axis=0)
    depth = np.reshape(depth, (1, img_height*img_width))
    ccoord_depth = np.dot(intrinsics_depth_inv, pcoord_depth) * depth
    ccoord_depth[1,:] = - ccoord_depth[1,:]
    ccoord_depth[2,:] = - ccoord_depth[2,:]
    ccoord_depth = np.concatenate((ccoord_depth, ones), axis=0)
    ccoord_color = np.dot(calibration_extrinsics, ccoord_depth)
    ccoord_color = ccoord_color[0:3,:]
    ccoord_color[1,:] = - ccoord_color[1,:]
    ccoord_color[2,:] = depth
    # ccoord_depth: depth camera coordinate
    # ccoord_color: color camera coordinate
    
    pcoord_color = np.dot(intrinsics_color, ccoord_color)   # project to image
    pcoord_color = pcoord_color[:,pcoord_color[2,:]!=0]     # eliminate (z==0)
    
    pcoord_color[0,:] = pcoord_color[0,:]/pcoord_color[2,:]+0.5
    pcoord_color[0,:] = pcoord_color[0,:].astype(int)
    pcoord_color[1,:] = pcoord_color[1,:]/pcoord_color[2,:]+0.5
    pcoord_color[1,:] = pcoord_color[1,:].astype(int)
    pcoord_color = pcoord_color[:,pcoord_color[0,:]>=0]
    pcoord_color = pcoord_color[:,pcoord_color[1,:]>=0]
    pcoord_color = pcoord_color[:,pcoord_color[0,:]<img_width]
    pcoord_color = pcoord_color[:,pcoord_color[1,:]<img_height]

    depth_[pcoord_color[1,:].astype(int),
           pcoord_color[0,:].astype(int)] = pcoord_color[2,:]
    #print('depth:', depthGG.shape)
    #print('depth_:', depth_.shape)
    return depth_