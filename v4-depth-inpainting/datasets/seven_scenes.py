from __future__ import division

import os
import random
import numpy as np
import cv2
from torch.utils import data

from .utils import *


class SevenScenes(data.Dataset):
    def __init__(self, root, dataset='7S', scene='heads', split='train', 
                    model='hscnet', aug='True'):
        self.intrinsics_color = np.array([[525.0, 0.0,     320.0],
                       [0.0,     525.0, 240.0],
                       [0.0,     0.0,  1.0]])

        self.intrinsics_depth = np.array([[585.0, 0.0,     320.0],
                       [0.0,     585.0, 240.0],
                       [0.0,     0.0,  1.0]])
        
        self.intrinsics_depth_inv = np.linalg.inv(self.intrinsics_depth)
        self.intrinsics_color_inv = np.linalg.inv(self.intrinsics_color)
        self.model = model
        self.dataset = dataset
        self.aug = aug 
        self.root = os.path.join(root,'7Scenes')
        self.calibration_extrinsics = np.loadtxt(os.path.join(self.root, 
                        'sensorTrans.txt'))
        self.scene = scene
        if self.dataset == '7S':
            self.scene_ctr = np.loadtxt(os.path.join(self.root, scene,
                            'translation.txt'))
        else: 
            self.scenes = ['chess','fire','heads','office','pumpkin',
                            'redkitchen','stairs']
            self.transl = [[0,0,0],[10,0,0],[-10,0,0],[0,10,0],[0,-10,0],
                            [0,0,10],[0,0,-10]]
            self.ids = [0,1,2,3,4,5,6]
            self.scene_data = {}
            for scene, t, d in zip(self.scenes, self.transl, self.ids):
                self.scene_data[scene] = (t, d, np.load(os.path.join(self.root,
                    scene,  'centers.npy')),
                    np.loadtxt(os.path.join(self.root, 
                    scene,'translation.txt')))

        self.split = split
        self.obj_suffixes = ['.color.png','.pose.txt','.depth.png']
        self.obj_keys = ['color','pose','depth']
                    
        with open(os.path.join(self.root, '{}{}'.format(self.split, 
                '.txt')), 'r') as f:
            self.frames = f.readlines()
            if self.dataset == '7S' or self.split == 'test':
                self.frames = [frame for frame in self.frames \
                if self.scene in frame]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index].rstrip('\n')
        scene, seq_id, frame_id = frame.split(' ') 

        obj_files = ['{}{}'.format(frame_id, 
                    obj_suffix) for obj_suffix in self.obj_suffixes]
        obj_files_full = [os.path.join(self.root, scene, 
                    seq_id, obj_file) for obj_file in obj_files]
        objs = {}
        for key, data in zip(self.obj_keys, obj_files_full):
            objs[key] = data
       
        depth = cv2.imread(objs['depth'],-1)
     
        depth[depth==65535] = 0
        
        mask = np.zeros_like(depth)
        mask[depth == 0] = 1
        mask = mask * 255
        mask = mask.astype(np.uint8)
        
        RADIOUS = 3
        depth_inpainting = cv2.inpaint(depth, mask, RADIOUS, cv2.INPAINT_NS)
        #print()
        cv2.imwrite(objs['color'][:-10]+'.depth_inpaint.png', depth_inpainting)
        
        return objs['color']
