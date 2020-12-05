# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 09:25:10 2020

@author: MGD
"""

import torch
import numpy as np


def detach_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(detach_hidden(v) for v in h)
    
def get_chunks(inp, t, k, chunk_size):    
    """ Split into chunks """
    
    idx = torch.arange(k * chunk_size , (k+1) * chunk_size)
    inp = inp[:,:,idx]
    t = t[:,idx]
    return inp.float(), t

def get_range(l, b):
    return range(1, l // b)

def model_performance(ground, pred, th):
    """ Custom way to quantify the performance of the peak detector"""
    
    TC, TP, FP, TC_p = 0, 0, 0, 0
    idx = np.arange(len(pred)-1)
    
    diff = np.diff(ground)
    a = diff > 0
    b = diff < 0
       
    if ground[0] == 1: a[0] = True
    if ground[-1] == 1: b[-1] = True
    if np.sum(a) > 0: 
        if np.sum(a) == np.sum(b):
            for l,n in zip(idx[a],idx[b]):
                if np.sum(pred[l:n]) / (n - l) > th:
                    TP += 1
                TC += 1
        else:
            print('Error during model performance check')
    
    diff = np.diff(pred)
    c = diff > 0
    d = diff < 0
    
    if pred[0] == 1: c[0] = True
    if pred[-1] == 1: d[-1] = True
    if np.sum(c) > 0: 
        if np.sum(c) == np.sum(d):
            for l,n in zip(idx[c],idx[d]):
                if np.sum(ground[l:n]) / (n - l) < th:
                    FP += 1
                TC_p += 1
        else:
            print('Error during model performance check')        
            
    return TP, TC, FP, TC_p