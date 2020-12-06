# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:00:45 2020

@author: MGD
"""

import numpy as np
from torch.utils.data import DataLoader
from dataloader import SyntheticData
import models 
from torch import nn,optim
from torch.utils.tensorboard import SummaryWriter
import torch
from utils import get_chunks, get_range
from sklearn.metrics import f1_score, precision_score, recall_score
from datetime import datetime
import json
import os


time = datetime.now()
time = time.strftime("%y%m%d%H%M")
dataset = SyntheticData('./data', 0.7)

batch_size = 1
input_channels = 4
kernel_size = 3
depth_step = 16
lr = 0.00007
num_epochs = 100
mini_batch = 512
weight = [1., 1.4]

os.mkdir(f'./logs/{time}')
config = {'kernel_size' : kernel_size, 'depth_step' : depth_step, 'batch_size' : batch_size, 'mini_batch' : mini_batch, 'Optimizer' : 'Adam' , 'lr' : lr, 'weight' : weight }
with open(f'logs/{time}/config.json', 'w') as outfile:
    json.dump(config, outfile)

dataloader = DataLoader(dataset, batch_size = batch_size)

net = models.UNET_1D(input_channels,depth_step,kernel_size)

writer = SummaryWriter(log_dir = f'./logs/{time}')

weight = torch.tensor(weight)

loss_fn = nn.NLLLoss(weight)
optimizer = optim.Adam(net.parameters(),lr = lr)

max_acc = 0

l,v = dataset.get_length()
num_batches = get_range(l, mini_batch)
val_batches = get_range(v,mini_batch)

for i in range(num_epochs):
    
    """ Set variables to zero """
    batch_losses , counter, batch_accuracy = 0 , 0, 0
    recall , precision, val_recall, val_precision = 0, 0, 0, 0
    epoch_losses , epoch_counter, epoch_accuracy = 0 , 0, 0
    
    for data in  dataloader:
        
        """ Training """
        net.train()
        targets = data['training']['labels']
        inputs = data['training']['sequence']
                         
        for k in num_batches:
            
            """ Get truncated steps """
            x,t = get_chunks(inputs,targets,k,mini_batch)
            
            if torch.sum(t) > 0 or np.random.uniform(0,1) < 0.1:
                outputs = net(x)
                pred = torch.max(outputs, dim = 1)[1].data.numpy()[0,:]
                ground = t.data.numpy()[0,:]
                
                optimizer.zero_grad()
                loss = loss_fn(outputs,t)
                
                loss.backward()
                optimizer.step()
                
                batch_losses += loss.data.numpy()
                batch_accuracy += f1_score(ground,pred,average = 'macro')
                recall += recall_score(ground,pred,average = 'macro')
                precision += precision_score(ground,pred,average = 'macro')
    
                counter += 1
                    
            
            if k % 400 == 0:               
                print(f'''Epoch {i}, chunk {k}: Training loss = {batch_losses / counter : .3f},  Training Accuracy = {batch_accuracy / counter : .3f},  Recall =  {recall / counter : .3f}, Precision =  {precision / counter: .3f}''')
                      
        
        """ Evaluation """
        targets = data['validation']['labels']
        inputs = data['validation']['sequence']
        
        net.eval()         
        for k in val_batches:
            
            """ Get truncated steps """
            x,t = get_chunks(inputs,targets,k,mini_batch)
            
            # if torch.sum(t) > 0 or np.random.uniform(0,1) < 0.1 :
            with torch.no_grad():
                outputs = net(x)
                pred = torch.max(outputs, dim = 1)[1].data.numpy()[0,:]
                ground = t.data.numpy()[0,:]
                
                loss = loss_fn(outputs,t)
                                    
                epoch_losses += loss.data.numpy()
                epoch_accuracy += f1_score(ground,pred,average = 'macro')
                val_recall += recall_score(ground,pred,average = 'macro')
                val_precision += precision_score(ground,pred,average = 'macro')
                epoch_counter += 1
                    

    
    """ Writting on the log """
    writer.add_scalars('Validation', {'val_loss' : epoch_losses / epoch_counter, 'val_acc' : epoch_accuracy / epoch_counter,
                                      'val_recall': val_recall / epoch_counter, 'val_precision': val_precision / epoch_counter }, i) 
    writer.add_scalars('Training', {'train_loss' : batch_losses / counter, 'train_acc' : batch_accuracy / counter,
                                      'train_recall': recall / counter, 'train_precision': precision / counter }, i)

    print(f'Training Epoch {i}: Loss = {batch_losses / counter : .3f},  Accuracy = {batch_accuracy / counter : .3f},  Recall =  {recall / counter : .3f}, Precision =  {precision / counter: .3f}')
    print(f'Validation Epoch {i}: Loss = {epoch_losses / epoch_counter : 3f},  Accuracy = {epoch_accuracy / epoch_counter : .3f}, Recall =  {val_recall / epoch_counter : .3f}, Precision =  {val_precision / epoch_counter: .3f}')
    
    """ Saving best model """
    if epoch_accuracy / epoch_counter > max_acc:
        max_acc = epoch_accuracy / epoch_counter
        torch.save(net,f'./logs/{time}/best_performance.pt')
        

writer.close()
torch.save(net,f'./logs/{time}/last_model.pt')
