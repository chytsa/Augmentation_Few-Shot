from __future__ import division

import sys
import os
import argparse
import numpy as np
import cv2
import torch
from torch.utils import data

sys.path.insert(0, './pnpransac')
from pnpransac import pnpransac
from models import get_model
from datasets import get_dataset

import time

def get_pose_err(pose_gt, pose_est):
    transl_err = np.linalg.norm(pose_gt[0:3,3]-pose_est[0:3,3])
    rot_err = pose_est[0:3,0:3].T.dot(pose_gt[0:3,0:3])
    rot_err = cv2.Rodrigues(rot_err)[0]
    rot_err = np.reshape(rot_err, (1,3))
    rot_err = np.reshape(np.linalg.norm(rot_err, axis = 1), -1) / np.pi * 180.
    return transl_err, rot_err[0]

def eval(args):
    scenes_7S = ['chess', 'fire', 'heads', 'office', 'pumpkin',
            'redkitchen','stairs']
    
    scenes_12S = ['apt1/kitchen', 'apt1/living', 'apt2/bed',
            'apt2/kitchen', 'apt2/living', 'apt2/luke', 
            'office1/gates362', 'office1/gates381', 
            'office1/lounge', 'office1/manolis',
            'office2/5a', 'office2/5b']
    
    scenes_Cambridge = ['GreatCourt', 'KingsCollege', 'OldHospital',
            'ShopFacade', 'StMarysChurch']

    if args.dataset in ['7S', 'i7S']:
        if args.scene not in scenes_7S:
            print('Selected scene is not valid.')
            sys.exit()

    if args.dataset in ['12S', 'i12S']:
        if args.scene not in scenes_12S:
            print('Selected scene is not valid.')
            sys.exit()

    if args.dataset == 'Cambridge':
        if args.scene not in scenes_Cambridge:
            print('Selected scene is not valid.')
            sys.exit()

    if args.dataset == 'i19S':
        if args.scene not in scenes_7S + scenes_12S:
            print('Selected scene is not valid.')
            sys.exit()

    # prepare datasets
    if args.dataset == 'i19S':
        datasetSs = get_dataset('7S')
        datasetTs = get_dataset('12S')
        if args.scene in scenes_7S: 
            datasetSs = datasetSs(args.data_path, args.dataset, args.scene,
                    split='test')
            datasetTs = datasetTs(args.data_path, args.dataset)
            dataset = datasetSs
        if args.scene in scenes_12S: 
            datasetSs = datasetSs(args.data_path, args.dataset)
            datasetTs = datasetTs(args.data_path, args.dataset, args.scene,
                    split='test')
            dataset = datasetTs
        centers = np.reshape(np.array([[]]),(-1,3))
        for scene in scenes_7S:
            centers = np.concatenate([centers, datasetSs.scene_data[scene][2]
                    + datasetSs.scene_data[scene][0]])
        for scene in scenes_12S:
            centers = np.concatenate([centers, datasetTs.scene_data[scene][2]
                    + datasetTs.scene_data[scene][0]])
    elif args.dataset == 'i7S':
        dataset = get_dataset('7S')
        dataset = dataset(args.data_path, args.dataset, args.scene, 
                split='test')
        centers = np.reshape(np.array([[]]),(-1,3))
        for scene in scenes_7S:
            centers = np.concatenate([centers, dataset.scene_data[scene][2]
                    + dataset.scene_data[scene][0]])
    elif args.dataset == 'i12S':
        dataset = get_dataset('12S')
        dataset = dataset(args.data_path, args.dataset, args.scene, 
                split='test')
        centers = np.reshape(np.array([[]]),(-1,3))
        for scene in scenes_12S:
            centers = np.concatenate([centers, dataset.scene_data[scene][2]
                    + dataset.scene_data[scene][0]])
    else:
        dataset = get_dataset(args.dataset)
        dataset = dataset(args.data_path, args.dataset, args.scene, 
                split='test')
        centers = dataset.centers
    intrinsics_color = dataset.intrinsics_color
    dataloader = data.DataLoader(dataset, batch_size=1,
                                  num_workers=4, shuffle=False)
    pose_solver = pnpransac(intrinsics_color[0,0], intrinsics_color[1,1],
            intrinsics_color[0,2], intrinsics_color[1,2])

    # prepare model
    torch.set_grad_enabled(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(args.model, args.dataset)
    model_state = torch.load(args.checkpoint, 
                map_location=device)['model_state']
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    # start evaluation
    rot_err_list = []
    transl_err_list = []

    x = np.linspace(4, 640-4, 80) + 106 * (args.dataset == 'Cambridge')
    y = np.linspace(4, 480-4, 60)
    xx, yy = np.meshgrid(x, y)
    pcoord = np.concatenate((np.expand_dims(xx,axis=2), 
            np.expand_dims(yy,axis=2)), axis=2)
    """
    if not os.path.exists(args.scene):
        os.mkdir(args.scene)
    if not os.path.exists(args.scene+"/label"):
        os.mkdir(args.scene+"/label")
    if not os.path.exists(args.scene+"/lbl_1"):
        os.mkdir(args.scene+"/lbl_1")
    if not os.path.exists(args.scene+"/lbl_2"):
        os.mkdir(args.scene+"/lbl_2")
    if not os.path.exists(args.scene+"/coord"):
        os.mkdir(args.scene+"/coord")
    if not os.path.exists(args.scene+"/score"):
        os.mkdir(args.scene+"/score")
    #if not os.path.exists("pose"):
    #    os.mkdir("pose")
    """
    total_times = []
    model_times = []
    ransac_times = []
    
    for _, (img_path, img, pose) in enumerate(dataloader):
        START = time.process_time()
        if args.dataset == 'Cambridge':
            img = img[:,:,:,106:106+640].to(device)
        else:
            img = img.to(device)
        if args.model == 'hscnet':
            coord, lbl_2, lbl_1 = model(img)
            """
            head, img_fname = os.path.split(img_path[0])
            img_fname, _ = os.path.splitext(img_fname)
            img_fname, _ = os.path.splitext(img_fname)
            img_fname = img_fname[-3:]
            #print(img_fname)
            
            _, seq = os.path.split(head)
            #print(seq)
            
            img_fname = seq + "-" + img_fname
            """
            # =======================================================
            lbl_1_vis = lbl_1.cpu().data.numpy().astype(np.float)
            lbl_2_vis = lbl_2.cpu().data.numpy().astype(np.float)
            
            #print(lbl_1_vis.shape)
            lbl_1_vis = np.transpose(lbl_1_vis, (0, 2, 3, 1))
            lbl_1_vis = np.reshape(lbl_1_vis, (60, 80, 25))
            #print(lbl_1_vis.shape)
            lbl_2_vis = np.transpose(lbl_2_vis, (0, 2, 3, 1))
            lbl_2_vis = np.reshape(lbl_2_vis, (60, 80, 25))
            """
            lbl_1_vis_fname = args.scene + "/lbl_1/" + img_fname + ".lbl_1"
            np.save(lbl_1_vis_fname, lbl_1_vis)
            lbl_2_vis_fname = args.scene + "/lbl_2/" + img_fname + ".lbl_2"
            np.save(lbl_2_vis_fname, lbl_2_vis)
            """
            # =======================================================
            lbl_1 = torch.argmax(lbl_1, dim=1)
            lbl_2 = torch.argmax(lbl_2, dim=1)
            lbl = (lbl_1 * 25 + lbl_2).cpu().data.numpy()[0,:,:]
            ctr_coord = centers[np.reshape(lbl,(-1)),:]
            ctr_coord = np.reshape(ctr_coord, (60,80,3))
            coord = np.transpose(coord.cpu().data.numpy()[0,:,:,:], (1,2,0))
            coord = coord + ctr_coord
            
            """
            # save label map and coord map
            lbl = lbl.astype(np.uint16)
            
            #print(img_path[0])
            #cv2.imwrite(fname, depth_new)
            
            lbl_fname = args.scene + "/label/" + img_fname + ".label.png"
            coord_fname = args.scene + "/coord/" + img_fname + ".coord"
            #print(lbl_fname)
            cv2.imwrite(lbl_fname, lbl)
            #print(coord.shape)
            np.save(coord_fname, coord)
            #print("pause")
            #input()
            """
        else:
            coord = np.transpose(model(img).cpu().data.numpy()[0,:,:,:], 
                    (1,2,0))
        
        END_MODEL = time.process_time()
        
        coord = np.ascontiguousarray(coord)
        pcoord = np.ascontiguousarray(pcoord)
        """
        f = open('imgname.txt', 'w')
        f.write(args.scene + "/score/" + img_fname + ".txt")
        f.close()
        """
        
        rot, transl = pose_solver.RANSAC_loop(np.reshape(pcoord, 
                (-1,2)).astype(np.float64), np.reshape(coord,  
                (-1,3)).astype(np.float64), args.num_hyp) 
        
        END_RANSAC = time.process_time()
        #print("total 執行時間：%f 秒" % (END_RANSAC - START))
        #print("model 執行時間：%f 秒" % (END_MODEL - START))
        #print("ransac執行時間：%f 秒" % (END_RANSAC - END_MODEL))
        total_times.append(END_RANSAC - START)
        model_times.append(END_MODEL - START)
        ransac_times.append(END_RANSAC - END_MODEL)
        
        pose_gt = pose.data.numpy()[0,:,:]
        pose_est = np.eye(4)
        pose_est[0:3,0:3] = cv2.Rodrigues(rot)[0].T
        pose_est[0:3,3] = -np.dot(pose_est[0:3,0:3], transl)
        """
        pose_fname = "pose/" + img_fname + ".pose.txt"
        #print(pose_fname)
        f = open(pose_fname, 'w')
        for i in range(pose_est.shape[0]):
            for j in range(pose_est.shape[1]):
                f.write(str(pose_est[i][j]))
                f.write(" ")
            f.write("\n")
        f.close()
        """
        
        
        transl_err, rot_err = get_pose_err(pose_gt, pose_est)

        rot_err_list.append(rot_err)
        transl_err_list.append(transl_err)
        
        #print('Pose error: {}m, {}\u00b0'.format(transl_err, rot_err)) 
    
    results = np.array([transl_err_list, rot_err_list]).T
    np.savetxt(os.path.join(args.output,
            'pose_err_{}_{}_{}.txt'.format(args.dataset, 
            args.scene.replace('/','.'), args.model)), results)
    if args.dataset != 'Cambridge':
        print('Accuracy: {}%'.format(np.sum((results[:,0] <= 0.05) 
                * (results[:,1] <= 5)) * 1. / len(results) * 100))
    print('Median pose error: {}m, {}\u00b0'.format(np.median(results[:,0]), 
            np.median(results[:,1])))
    
    print("total  time:", np.average(total_times))
    print("model  time:", np.average(model_times))
    print("ransac time:", np.average(ransac_times))
    
    # RANSAC time
    a = []
    b = []
    c = []
    with open('result.txt') as f:
        for line in f.readlines():
            s = line.rstrip('\n').split(' ')
            #print(s)
            a.append(int(s[0]))
            b.append(float(s[1]))
            c.append(float(s[2]))
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)

    #print(a.shape)

    print("loop:", np.mean(a))
    print("Step1 time:", np.mean(b))
    print("Step2 time:", np.mean(c))
    print("")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hscnet")
    parser.add_argument('--model', nargs='?', type=str, default='hscnet',
                        choices=('hscnet', 'scrnet'),
                        help='Model to use [\'hscnet, scrnet\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='7S', 
                        choices=('7S', '12S', 'i7S', 'i12S', 'i19S',
                        'Cambridge'), help='Dataset to use')
    parser.add_argument('--scene', nargs='?', type=str, default='heads', 
                        help='Scene')
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to saved model')
    parser.add_argument('--data_path', required=True, type=str, 
                        help='Path to dataset')
    parser.add_argument('--output', nargs='?', type=str, default='./',
                        help='Output directory')
    parser.add_argument('--num_hyp', nargs='?', type=int, default=256,
                        help='number of hypotheses')
    args = parser.parse_args()
    print("Number of hypotheses:", args.num_hyp)
    eval(args)
