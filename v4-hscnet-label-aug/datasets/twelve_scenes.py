from __future__ import division

import os
import random
import numpy as np
import cv2
from torch.utils import data

from .utils import *


class TwelveScenes(data.Dataset):
    def __init__(self, root, dataset='12S', scene='apt2/bed', split='train', 
                    model='hscnet', aug='True', scene_txt_postfix=''):
        self.intrinsics_color = np.array([[572.0, 0.0,     320.0],
                       [0.0,     572.0, 240.0],
                       [0.0,     0.0,  1.0]])
                       
        self.intrinsics_color_inv = np.linalg.inv(self.intrinsics_color)
        
        self.model = model
        self.dataset = dataset
        self.aug = aug 
        self.root = os.path.join(root,'12Scenes')
        self.scene = scene
        if self.dataset == '12S':
            self.centers = np.load(os.path.join(self.root, scene,
                            'centers.npy'))
        else: 
            self.scenes = ['apt1/kitchen','apt1/living','apt2/bed',
                    'apt2/kitchen','apt2/living','apt2/luke','office1/gates362',
                    'office1/gates381','office1/lounge','office1/manolis',
                    'office2/5a','office2/5b']
            self.transl = [[0,-20,0],[0,-20,0],[20,0,0],[20,0,0],[25,0,0],
                    [20,0,0],[-20,0,0],[-25,5,0],[-20,0,0],[-20,-5,0],[0,20,0],
                    [0,20,0]]
            if self.dataset == 'i12S':
                self.ids = [0,1,2,3,4,5,6,7,8,9,10,11]
            else:
                self.ids = [7,8,9,10,11,12,13,14,15,16,17,18]
            self.scene_data = {}
            for scene, t, d in zip(self.scenes, self.transl, self.ids):
                self.scene_data[scene] = (t, d, np.load(os.path.join(self.root,
                    scene,  'centers.npy')))

        self.split = split
        self.obj_suffixes = ['.color.png', '.pose.txt', '.depth.png']
        self.obj_keys = ['color', 'pose', 'depth']
        
        if self.dataset == '12S' or self.split == 'test':
            with open(os.path.join(self.root, self.scene, 
                    '{}{}{}'.format("aug_32", scene_txt_postfix, '.txt')), 'r') as f:
                self.frames = f.readlines()
        else:
            self.frames = []
            for scene in self.scenes:
                with open(os.path.join(self.root, scene, 
                        '{}{}'.format(self.split, '.txt')), 'r') as f:
                    frames = f.readlines()
                    frames = [scene + ' ' + frame for frame in frames ]
                self.frames.extend(frames)

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        frame = self.frames[index].rstrip('\n')

        if self.dataset != '12S' and self.split == 'train':
            scene, frame = frame.split(' ')
            centers = self.scene_data[scene][2] 
        else: 
            scene = self.scene
            if self.split == 'train':
                centers = self.centers
        
        obj_files = ['{}{}'.format(frame, 
                    obj_suffix) for obj_suffix in self.obj_suffixes]
        obj_files_full = [os.path.join(self.root, scene, 
                    'train_aug_0.5-20_cali-set-point-size-2.0', obj_file) for obj_file in obj_files]
        objs = {}
        for key, data in zip(self.obj_keys, obj_files_full):
            objs[key] = data
        
        img = cv2.imread(objs['color'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480))

        pose = np.loadtxt(objs['pose'])
        
        #ctr_coord = centers[np.reshape(lbl,(-1))-1,:]
        #ctr_coord = np.reshape(ctr_coord,(480,640,3)) * 1000

        depth = cv2.imread(objs['depth'],-1)
        
        pose[0:3,3] = pose[0:3,3] * 1000
        
        depth[depth==65535] = 0
        depth = depth * 1.0
        
        coord, mask = get_coord(depth, pose, self.intrinsics_color_inv)

        label = np.zeros_like(depth)
        centers = centers.astype(np.float64)
        centers *= 1000
        
        for i in range(coord.shape[0]):
            for j in range(coord.shape[1]):
                x = np.tile(coord[i, j, :], (centers.shape[0], 1))
                dis = np.linalg.norm(x - centers, axis=1)
                label[i, j] = np.argmin(dis) + 1
        
        label = label.astype(np.uint16)
        
        head, tail = os.path.split(objs['color'])
        filename, file_extension = os.path.splitext(tail)
        filename, file_extension = os.path.splitext(filename)
        fname = filename + '.label.png'
        fname = os.path.join(head, fname)
        #print(fname)
        cv2.imwrite(fname, label)
        
        return objs['color']
        
