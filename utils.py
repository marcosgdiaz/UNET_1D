# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 09:25:10 2020

@author: MGD
"""

import torch


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