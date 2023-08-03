from __future__ import division

import os
import torch

def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')

def adjust_lr(optimizer, init_lr, c_iter, n_iter):
    if (c_iter + np.ceil(200000/batch_size) - n_iter) >= 0:
        zz = (c_iter + np.ceil(200000/batch_size) - n_iter) // np.ceil(50000/batch_size) + 1
    else:
        zz = 0

    lr = init_lr * (0.5 ** zz)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr    

def save_state(savepath, epoch, model, optimizer):
    state = {'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()}
    filepath = os.path.join(savepath, 'model_'+str(epoch)+'.pkl')
    torch.save(state, filepath)