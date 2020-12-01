# -*- coding: utf-8 -*-

"""
Created on Mon Nov  9 15:49:13 2020

@author: MGD
"""

import torch
from torch.utils.data import Dataset
from os import listdir, path
from pandas import read_csv

class SyntheticData(Dataset):
    
    """ Sythetic dataset for peak detection """
    
    def __init__(self, root, split):
        try:
            filenames = listdir(root)
        except:
            print('The path to the synthetic data does not exist')
            return None
        
        self.split = split
        self.root_dir = root
        self.csv_files = [ filename for filename in filenames if filename.endswith(".csv")]
        if len(self.csv_files) < 1:
            print('No csv files were found in the path')
            return None
        
        
    def __len__(self):
        return len(self.csv_files) 
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        csv_name = self.csv_files[idx]
        data = read_csv(path.join(self.root_dir, csv_name),  header = None)
        time = data.iloc[:,0].values
        sequence = data.iloc[:,1:5].values.transpose()
        labels = data.iloc[:,5].values
        s = int(sequence.shape[1] * self.split)
        
        training_sample = {'time' : time[:s], 'sequence' : sequence[:,:s], 'labels' : labels[:s]}
        validation_sample = {'time' : time[s:], 'sequence' : sequence[:,s:], 'labels' : labels[s:]}
        
        return {'training' : training_sample, 'validation' : validation_sample}
    
    def get_length(self):
        csv_name = self.csv_files[0]
        data = read_csv(path.join(self.root_dir, csv_name),  header = None)
        return (int(data.shape[0]*self.split),int(data.shape[0]*(1-self.split)))
    
    
def collate_fn(batch):
    data_list, label_list = [], []
    for data in batch:
        data_list.append(data['sequence'])
        label_list.append(data['labels'])
    return torch.Tensor(data_list), torch.LongTensor(label_list)
        
