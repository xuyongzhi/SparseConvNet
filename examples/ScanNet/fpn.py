# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Options
m = 16 # 16 or 32
residual_blocks=False #True or False
block_reps = 1 #Conv block repetition factor: 1 or 2

import torch, data, iou
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn
import time
import os, sys, glob
import math
import numpy as np

use_cuda = torch.cuda.is_available()
exp_name='results/unet_scale20_m16_rep1_notResidualBlocks'


nPlanes = [m, 2*m, 3*m, 4*m]
unet = scn.FPN_Net(data.full_scale, data.dimension, block_reps, nPlanes, residual_blocks)
if use_cuda:
    unet=unet.cuda()

training_epochs=512
training_epoch=scn.checkpoint_restore(unet,exp_name,'unet',use_cuda)
optimizer = optim.Adam(unet.parameters())
print('#classifer parameters', sum([x.nelement() for x in unet.parameters()]))

for epoch in range(training_epoch, training_epochs+1):
    unet.train()
    stats = {}
    scn.forward_pass_multiplyAdd_count=0
    scn.forward_pass_hidden_states=0
    start = time.time()
    train_loss=0
    for i,batch in enumerate(data.train_data_loader):
        optimizer.zero_grad()
        if use_cuda:
            batch['x'][1]=batch['x'][1].cuda()
            batch['y']=batch['y'].cuda()
        predictions=unet(batch['x'])
        loss = torch.nn.functional.cross_entropy(predictions,batch['y'])
        train_loss+=loss.item()
        loss.backward()
        optimizer.step()
    print(epoch,'Train loss',train_loss/(i+1), 'MegaMulAdd=',scn.forward_pass_multiplyAdd_count/len(data.train)/1e6, 'MegaHidden',scn.forward_pass_hidden_states/len(data.train)/1e6,'time=',time.time() - start,'s')
    scn.checkpoint_save(unet,exp_name,'unet',epoch, use_cuda)

    if scn.is_power2(epoch):
        with torch.no_grad():
            unet.eval()
            store=torch.zeros(data.valOffsets[-1],20)
            scn.forward_pass_multiplyAdd_count=0
            scn.forward_pass_hidden_states=0
            start = time.time()
            for rep in range(1,1+data.val_reps):
                for i,batch in enumerate(data.val_data_loader):
                    if use_cuda:
                        batch['x'][1]=batch['x'][1].cuda()
                        batch['y']=batch['y'].cuda()
                    predictions=unet(batch['x'])
                    store.index_add_(0,batch['point_ids'],predictions.cpu())
                print(epoch,rep,'Val MegaMulAdd=',scn.forward_pass_multiplyAdd_count/len(data.val)/1e6, 'MegaHidden',scn.forward_pass_hidden_states/len(data.val)/1e6,'time=',time.time() - start,'s')
                iou.evaluate(store.max(1)[1].numpy(),data.valLabels)