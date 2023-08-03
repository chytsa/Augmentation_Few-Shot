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
        self.obj_suffixes = ['.color.png','.pose.txt','.depth_inpaint.png']
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

        if self.dataset!='7S':
            centers = self.scene_data[scene][2] 
            scene_ctr = self.scene_data[scene][3] 
        else:
            scene_ctr = self.scene_ctr   

        obj_files = ['{}{}'.format(frame_id, 
                    obj_suffix) for obj_suffix in self.obj_suffixes]
        obj_files_full = [os.path.join(self.root, scene, 
                    seq_id, obj_file) for obj_file in obj_files]
        objs = {}
        for key, data in zip(self.obj_keys, obj_files_full):
            objs[key] = data
       
        img = cv2.imread(objs['color'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pose = np.loadtxt(objs['pose'])

        #pose[0:3,3] = pose[0:3,3] - scene_ctr
        
        if self.dataset != '7S' and (self.model != 'hscnet' \
                        or self.split == 'test'):
            pose[0:3,3] = pose[0:3,3] + np.array(self.scene_data[scene][0])
        
        depth = cv2.imread(objs['depth'],-1)
        
        pose[0:3,3] = pose[0:3,3] * 1000
     
        depth[depth==65535] = 0
        depth = depth * 1.0
        depth = get_depth(depth, self.calibration_extrinsics, 
            self.intrinsics_color, self.intrinsics_depth_inv)
        
        depth = depth.astype(np.uint16)
        head, tail = os.path.split(objs['color'])
        
        _, seq = os.path.split(head)
        
        filename, file_extension = os.path.splitext(tail)
        filename, file_extension = os.path.splitext(filename)
        #filename, file_extension = os.path.splitext(objs['color'])
        #print(str(int(filename[-3:]))+'.png')
        #print()
        
        if not os.path.exists(seq):
            os.mkdir(seq)
        
        fname = str(int(filename[-3:]))+'.png'
        fname = os.path.join(seq, fname)
        
        cv2.imwrite(fname, depth)
        #print("=====")
        #coord, mask = get_coord(depth, pose, self.intrinsics_color_inv)
        return objs['color']
