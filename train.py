# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:00:45 2020

@author: MGD
"""

from torch.utils.data import DataLoader
from dataloader import SyntheticData
import models 
from torch import nn,optim
import torch
from utils import get_chunks, get_range
from sklearn.metrics import f1_score, precision_score, recall_score

dataset = SyntheticData('./data', 0.7)

batch_size = 1
dataloader = DataLoader(dataset, batch_size = batch_size)

net = models.UNET_1D(4,16,3)
# net = net.float()

num_epochs = 100
mini_batch = 512


weight = torch.tensor([1., 5.])

loss_fn = nn.NLLLoss(weight)
optimizer = optim.Adam(net.parameters(),lr = 0.00008)

losses, accuracy = [], []
batch_losses , counter, batch_accuracy = 0 , 0, 0
recall , precision = 0,0
epoch_losses , epoch_counter, epoch_accuracy = 0 , 0, 0
max_acc = 0

l,v = dataset.get_length()
num_batches = get_range(l, mini_batch)
val_batches = get_range(v,mini_batch)

for i in range(num_epochs):
    for data in  dataloader:
        
        """ Training """
        net.train()
        targets = data['training']['labels']
        inputs = data['training']['sequence']
                         
        for k in num_batches:
            
            """ Get truncated steps """
            x,t = get_chunks(inputs,targets,k,mini_batch)
            
            if torch.sum(t) > 0:
                outputs = net(x)
                pred = torch.max(outputs, dim = 1)[1].data.numpy()
    
                optimizer.zero_grad()
                loss = loss_fn(outputs,t)
                
                loss.backward()
                optimizer.step()
                
                batch_losses += loss.data.numpy()
                batch_accuracy += f1_score(t.data.numpy()[0,:],pred[0,:])
                recall += recall_score(t.data.numpy()[0,:],pred[0,:])
                precision += precision_score(t.data.numpy()[0,:],pred[0,:])
    
                counter += 1
                    
            
            if k % 200 == 0:               
                print(f'''Epoch {i}, chunk {k}: Training loss = {batch_losses / counter : .3f},  
                      Training Accuracy = {batch_accuracy / counter : .3f},  Recall =  {recall / counter : .3f}, Precision =  {precision / counter: .3f}''')
                losses.append( batch_losses / counter)
                accuracy.append( batch_accuracy / counter)
                batch_losses , counter, batch_accuracy = 0 , 0 , 0
                recall , precision = 0,0
                
        
        """ Evaluation """
        targets = data['validation']['labels']
        inputs = data['validation']['sequence']
        
                 
        for k in val_batches:
            
            """ Get truncated steps """
            x,t = get_chunks(inputs,targets,k,mini_batch)
            
            if torch.sum(t) > 0:
                with torch.no_grad():
                    outputs = net(x)
                    pred = torch.max(outputs, dim = 1)[1].data.numpy()
                    loss = loss_fn(outputs,t)
                    
                    epoch_losses += loss.data.numpy()
                    epoch_accuracy += f1_score(t.data.numpy()[0,:],pred[0,:])
                    epoch_counter += 1
            
                
    print(f'Epoch {i}: Validation loss = {epoch_losses / epoch_counter},  Validation Accuracy = {epoch_accuracy / epoch_counter}')
    if epoch_accuracy / epoch_counter > max_acc:
        max_acc = epoch_accuracy / epoch_counter
        torch.save(net,'./trained_models/trained_model_best.pt')
    epoch_losses , epoch_counter, epoch_accuracy = 0 , 0, 0
                
torch.save(net,'./trained_models/trained_model_last.pt')
