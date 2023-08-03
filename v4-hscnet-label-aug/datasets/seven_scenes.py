from __future__ import division

import os
import random
import numpy as np
import cv2
from torch.utils import data

from .utils import *


class SevenScenes(data.Dataset):
    def __init__(self, root, dataset='7S', scene='heads', 
                model='hscnet', scene_txt_postfix=''):
        self.intrinsics_color = np.array([[525.0, 0.0,     320.0],
                       [0.0,     525.0, 240.0],
                       [0.0,     0.0,  1.0]])

        self.intrinsics_depth = np.array([[525.0, 0.0,     320.0],
                       [0.0,     525.0, 240.0],
                       [0.0,     0.0,  1.0]])
        
        self.intrinsics_depth_inv = np.linalg.inv(self.intrinsics_depth)
        self.intrinsics_color_inv = np.linalg.inv(self.intrinsics_color)
        self.model = model
        self.dataset = dataset
        self.root = os.path.join(root,'7Scenes')
        #self.calibration_extrinsics = np.loadtxt(os.path.join(self.root, 
        #                'sensorTrans.txt'))
        #self.calibration_extrinsics = np.loadtxt(os.path.join(self.root, 
        #                'open3d.txt'))
        self.scene = scene
        self.scene_ctr = np.loadtxt(os.path.join(self.root, scene,
                            'translation.txt'))
        self.centers = np.load(os.path.join(self.root, scene,
                            'centers.npy'))
        self.obj_suffixes = ['.color.png','.pose.txt','.depth.png']
        self.obj_keys = ['color','pose','depth']
                    
        with open(os.path.join(self.root, scene, '{}{}{}'.format(scene, 
                scene_txt_postfix, '.txt')), 'r') as f:
            self.frames = f.readlines()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index].rstrip('\n')
        #scene, seq_id, frame_id = frame.split(' ')

        centers = self.centers
        scene_ctr = self.scene_ctr

        obj_files = ['{}{}'.format(frame, 
                    obj_suffix) for obj_suffix in self.obj_suffixes]
        obj_files_full = [os.path.join(self.root, self.scene, 
            'train_aug_0.5-64-k4_cali-set-point-size-2.0', obj_file) for obj_file in obj_files]
        objs = {}
        for key, data in zip(self.obj_keys, obj_files_full):
            objs[key] = data
        
        img = cv2.imread(objs['color'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        pose = np.loadtxt(objs['pose'])
        pose[0:3,3] = pose[0:3,3] - scene_ctr       
        
        #ctr_coord = centers[np.reshape(lbl,(-1))-1,:]
        #ctr_coord = np.reshape(ctr_coord,(480,640,3)) * 1000
        
        depth = cv2.imread(objs['depth'],-1)
        
        pose[0:3,3] = pose[0:3,3] * 1000
     
        depth[depth==65535] = 0
        depth = depth * 1.0
        #depth_cali = get_depth(depth, self.calibration_extrinsics, 
        #    self.intrinsics_color, self.intrinsics_depth_inv)
        #print("===============")
        #print(depth[200:220, 200:220])
        #print("***")
        #print(depth_cali[200:220, 200:220])
        #print("===============")
        coord, mask = get_coord(depth, pose, self.intrinsics_color_inv)
        
        label = np.zeros_like(depth)
        centers = centers.astype(np.float64)
        centers *= 1000
        
        for i in range(coord.shape[0]):
            for j in range(coord.shape[1]):
                x = np.tile(coord[i, j, :], (centers.shape[0], 1))
                dis = np.linalg.norm(x - centers, axis=1)
                label[i, j] = np.argmin(dis) + 1
        """
        coord_p = coord.reshape((-1, 3))
        num = coord_p.shape[0]
        coord_p = np.repeat(coord_p[:, np.newaxis, :], centers.shape[0], axis=1)
        
        centers_p = np.repeat(centers[np.newaxis, :, :], num, axis=0)
        
        dis = np.linalg.norm(coord_p - centers_p, axis=2)
        label = np.argmin(dis, axis=1)
        label = label + 1
        """
        label = label.astype(np.uint16)
        
        head, tail = os.path.split(objs['color'])
        filename, file_extension = os.path.splitext(tail)
        filename, file_extension = os.path.splitext(filename)
        fname = filename + '.label.png'
        fname = os.path.join(head, fname)
        #print(fname)
        cv2.imwrite(fname, label)
        
        #print(objs['color'])
        # ===================================================
        #if self.model == 'hscnet':
        #    coord = coord - ctr_coord
    
        #coord = coord[4::8,4::8,:]
        #mask = mask[4::8,4::8].astype(np.float16)
        #lbl = lbl[4::8,4::8].astype(np.float16)
        # ===================================================
        #mask = mask.astype(np.float16)
        
        #img, coord, mask = to_tensor(img, coord, mask)

        return objs['color']
