# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 19:10:58 2020

@author: MGD
"""

from torch import nn
import torch


class conv_step(nn.Module):
    def __init__(self, input_dim, num_features, kernel_size,p):
        super(conv_step, self).__init__()
        
        padding = int((kernel_size - 1) / 2)
        self.conv_1 = nn.Conv1d(input_dim,num_features, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(num_features)
        self.conv_2 = nn.Conv1d(num_features,num_features, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        
        self.dropout = nn.Dropout(p)
        
    def forward(self,x):
        x = self.conv_1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

class up_conv(nn.Module):
    def __init__(self, input_dim):
        super(up_conv, self).__init__()
        self.upsample = nn.Upsample(scale_factor = 2)
        self.conv = nn.Conv1d(input_dim, input_dim // 2,3,padding=1)
        self.bn = nn.BatchNorm1d(input_dim // 2)
        
    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        # x = self.bn(x)
        return x
    
class UNET_1D(nn.Module):
    def __init__(self ,input_dim,num_features,kernel_size,p):
        super(UNET_1D, self).__init__()
        
        self.down_layer_1 = conv_step(input_dim, num_features, kernel_size,p)
        self.down_layer_2 = conv_step(num_features, num_features*2, kernel_size,p)
        self.down_layer_3 = conv_step(num_features*2, num_features*4, kernel_size,p)
        self.down_layer_4 = conv_step(num_features*4, num_features*8, kernel_size,p)
        
        self.up_conv_1 = up_conv(num_features*8)
        self.up_conv_2 = up_conv(num_features*4)
        self.up_conv_3 = up_conv(num_features*2)
        
        self.up_layer_1 = conv_step(num_features*8, num_features*4,kernel_size)
        self.up_layer_2 = conv_step(num_features*4, num_features*2,kernel_size)
        self.up_layer_3 = conv_step(num_features*2, num_features, kernel_size)
        
        self.maxpool = nn.MaxPool1d(2,2,padding=0)
        self.softmax = nn.LogSoftmax(dim = 1)
        
        self.final = nn.Conv1d(num_features,2,1)
        
        self.bn = nn.BatchNorm1d(input_dim)
      
        
    def forward(self,x):
        
        #x = self.bn(x)

        
        """ Contracting """
        out_1 = self.down_layer_1(x)
        x = self.maxpool(out_1)
        
        out_2 = self.down_layer_2(x)
        x = self.maxpool(out_2)
        
        out_3 = self.down_layer_3(x)
        x = self.maxpool(out_3)
        
        end = self.down_layer_4(x)
        

        
        """ Expanding """
        x = self.up_conv_1(end)
        x = torch.cat([out_3,x],dim = 1)
        x = self.up_layer_1(x)
        

        
        x = self.up_conv_2(x)
        x = torch.cat([out_2,x],dim = 1)
        x = self.up_layer_2(x)
        

        
        x = self.up_conv_3(x)
        x = torch.cat([out_1,x],dim = 1)
        x = self.up_layer_3(x)
        
        x = self.final(x)
        
        return self.softmax(x)
    
        