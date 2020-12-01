# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:15:03 2020

@author: MGD
"""

import torch
from utils import get_chunks, get_range
from dataloader import SyntheticData
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score


net = torch.load('./trained_models/trained_model_best.pt')

dataset = SyntheticData('./data',0.7)
dataloader = DataLoader(dataset, batch_size = 1)

net.eval()
mini_batch = 512

l,v = dataset.get_length()
num_batches = get_range(l, mini_batch)


for i,data in enumerate(dataloader):
    
    targets = data['training']['labels']
    inputs = data['training']['sequence']
    
    epoch_losses , epoch_counter, epoch_accuracy = 0 , 0, 0
    for k in num_batches:
        x,t = get_chunks(inputs,targets,k,mini_batch)
        
        outputs = net(x)
        pred = torch.max(outputs, dim = 1)[1].data.numpy()
        
        epoch_accuracy += f1_score(t.data.numpy()[0,:],pred[0,:])
        epoch_counter += 1
    
    
        
    print(f'Index {i}: Validation Accuracy = {epoch_accuracy / epoch_counter}')
    
idx = torch.arange(20000,30000)
outputs = net(inputs[:,:,idx].float())
pred = torch.max(outputs, dim = 1)[1].data.numpy()
plt.figure()
plt.plot(targets[0,idx])
plt.plot(pred[0,:])
plt.figure()
plt.plot(targets[0,idx])
plt.plot(10**outputs[0,1,:].data.numpy())
# plt.plot(outputs[0,0,:].data.numpy())
#plt.plot(pred[0,:])
plt.figure()
plt.plot(inputs[0,0,idx])
plt.plot(inputs[0,1,idx])
plt.plot(inputs[0,2,idx])
plt.plot(inputs[0,3,idx])
plt.plot(pred[0,:]*torch.max(inputs[:,:,idx]).numpy())
