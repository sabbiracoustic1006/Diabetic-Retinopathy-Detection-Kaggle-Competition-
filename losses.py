#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 02:35:53 2019

@author: ratul
"""
import torch
import numpy as np
from fastai.layers import MSELossFlat
from torch import nn

device = 'cuda'

class customLoss(nn.Module):
    def __init__(self,alpha):
        super(customLoss,self).__init__()
        self.alpha = alpha
        self.beta = 1. - alpha
        self.xentropy = nn.CrossEntropyLoss()
        self.l1Loss = nn.L1Loss()
        
        
    def forward(self, pred, label, predReg, labelReg):
        xentropy = self.xentropy(pred,label)
        l1 = self.l1Loss(predReg,labelReg)
        
        loss = self.alpha*xentropy + self.beta*l1
        return loss
    
class myLossAnother(nn.Module):
    def __init__(self,reduction='mean'):
        super(myLossAnother,self).__init__()
        self.reduction = reduction
        
    def forward(self,y_pred,y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.float()
        diff = y_true-y_pred
        idxe = diff.abs() <= 1 
        idx4 = diff.abs() > 1
        loss = (torch.exp(diff.abs())-1.).mean()
#         torch.pow(diff[idx4],2).mean() + (
        return loss

class kapjhapLoss(nn.Module):
    def __init__(self):
        super(kapjhapLoss,self).__init__()
        self.l2 = MSELossFlat()
        self.mainfeature = torch.tensor(np.load('embedding_final.npy')).float().cuda()
        self.cosineloss = nn.CosineSimilarity(dim=2)
        
        
    def forward(self,y_pred,y_true):
        ypred,yFeat = y_pred
        mse = self.l2(ypred,y_true)
        bs = len(y_true)
        cosDis = self.cosineloss(yFeat.unsqueeze(1).repeat(1,5,1),
                                 self.mainfeature.unsqueeze(0).repeat(bs,1,1))
        
        a = torch.arange(bs).long().cuda()
        
        weight = 0.25*torch.ones(bs,5).float().cuda()
        
        weight[a,y_true.long()] = 1.
        cosDis[a,y_true.long()] = 1. - cosDis[a,y_true.long()]
        
        loss = (cosDis*weight).abs().sum()/bs + mse
        return loss

class qappa_loss(nn.Module):
    def __init__(self,y_pow=2,eps=1e-12,N=5):
        super(qappa_loss,self).__init__()
        self.y_pow = y_pow
        self.eps = eps
        self.N = N
        self.repeat_op = torch.arange(N).repeat(N,1).cuda().float()
        self.weights = torch.pow(self.repeat_op - self.repeat_op.t(),2)/torch.as_tensor((N-1)**2).float()
        self.xentropy = nn.CrossEntropyLoss()
        
        
    def forward(self,y_pred,y_true):
        y_one_hot = torch.zeros_like(y_pred)
        y_one_hot.scatter_(1, y_true.view(-1,1).long(), 1)
        
        pred_ = y_pred**self.y_pow
        pp = self.eps + pred_.sum(1).view(-1,1)
        
        pred_norm = pred_ / (self.eps + pred_.sum(1).view(-1,1))
        
        hist_rater_a = pred_norm.sum(0)
        hist_rater_b = y_one_hot.sum(0)
        
        conf_mat = pred_norm.t().mm(y_one_hot)
        
        num = torch.sum(self.weights*conf_mat)
        den = torch.sum(self.weights*torch.matmul(hist_rater_a.view(self.N,1),
                                             hist_rater_b.view(1,self.N)))/len(y_true)
        
        return num/(den+self.eps) + self.xentropy(y_pred,y_true.long())

class myLoss(nn.Module):
    def __init__(self,reduction='mean'):
        super(myLoss,self).__init__()
        self.reduction = reduction
        
    def forward(self,y_pred,y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.float()
        diff = y_true-y_pred
        sign = torch.sign(diff)
        loss = torch.log(torch.ones_like(diff) + diff*sign)
        return loss.mean() if self.reduction=='mean' else loss.sum()
    
    
class kapjhapMse(nn.Module):
    def __init__(self,cls_pref):
        super(kapjhapMse,self).__init__()
        self.cls_pref = int(cls_pref)
        
    def forward(self,y_pred,y_true):
        y_pred = y_pred.view(-1)
        idx = y_true == self.cls_pref
        weight = idx.clone().float()
        
        sqr_dif = (y_pred - y_true)**2
        weight[idx] = sqr_dif[idx] + 1.
        
        weight[idx==0] = 1.    
        loss = (weight*sqr_dif).mean()
        
        return loss
    
class EmbeddingLoss(nn.Module):
    def __init__(self):
        super(EmbeddingLoss,self).__init__()
        self.cos_dis = nn.CosineEmbeddingLoss()
        
    def forward(self,xs,y_true):
        _,x = xs
        
        bs = x.shape[0]
        
        idx = torch.combinations(torch.range(0,bs-1),2).long().cuda()
        
        x1 = x[idx[:,0]]
        x2 = x[idx[:,1]]
        
        y1 = y_true[idx[:,0]]
        y2 = y_true[idx[:,1]]
        
        y = 2*((y1 == y2).float()) - 1
        
        loss = self.cos_dis(x1,x2,y)
        return loss
