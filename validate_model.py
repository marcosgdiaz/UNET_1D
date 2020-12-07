# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:15:03 2020

@author: MGD
"""

import torch
from utils import get_chunks, get_range, model_performance
from dataloader import SyntheticData
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score,precision_score, recall_score
import json
from os.path import dirname

path = 'logs/2012062341/best_performance.pt'
root = dirname(path)
net = torch.load(path)
net.eval()

dataset = SyntheticData('./test_data',1)
dataloader = DataLoader(dataset, batch_size = 1)

mini_batch = 512

l,v = dataset.get_length()
num_batches = get_range(l, mini_batch)

N = len(dataset)

test_results = {'PNR' : [], 'Counts' : [], 'TP' : [], 'TC' : [] , 'FP' : []}
for i,data in enumerate(dataloader):
    
    targets = data['training']['labels']
    inputs = data['training']['sequence']
    counts, ground = 0, 0
    epoch_losses , epoch_counter, epoch_accuracy, recall, precision = 0, 0, 0, 0, 0
    true_positive, total_count, false_positive, total_count_p = 0, 0,0,0
    for k in num_batches:
        with torch.no_grad():        
            x,t = get_chunks(inputs,targets,k,mini_batch)
        
            outputs = net(x)
            pred = torch.max(outputs, dim = 1)[1].data.numpy()[0,:]
            ground = t.data.numpy()[0,:]
            
            TP, TC, FP, TC_p = model_performance(ground,pred,0.3)

            true_positive += TP
            total_count += TC
            false_positive += FP
            total_count_p += TC_p
            
            epoch_accuracy += f1_score(ground,pred,average = 'macro')
            recall += recall_score(ground,pred, average = 'macro')
            precision += precision_score(ground,pred, average = 'macro')
            epoch_counter += 1
    
    SNR = int(dataset.csv_files[i][0:-4].split('_')[1])
    Counts = int(dataset.csv_files[i][0:-4].split('_')[3])
    test_results['TP'].append(true_positive)
    test_results['TC'].append(total_count)
    test_results['FP'].append(false_positive)
    test_results['PNR'].append(SNR)
    test_results['Counts'].append(Counts)
     
    print(f'Index {i}: Validation Accuracy = {epoch_accuracy / epoch_counter : 3f}  Recall =  {recall / epoch_counter : .3f}, Precision =  {precision / epoch_counter: .3f}')
    print(f'Index {i}: True Peaks detected = {true_positive}  Total Expected Count =  {total_count}, False Peaks =  {false_positive},  Total_predicted =  {total_count_p}')

with open(f'{root}/test_result.json', 'w') as outfile:
    json.dump(test_results, outfile)
