#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:48:30 2019

@author: ratul
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.model_selection import StratifiedKFold
from joblib import load, dump
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from fastai import *
from fastai.vision import *
from fastai.callbacks import *
from torchvision import models as md
from torch import nn
from torch.nn import functional as F
import re
import math
import collections
from functools import partial
from torch.utils import model_zoo
from sklearn import metrics
from collections import Counter
import json
from qk import quadratic_weighted_kappa as qappa
from radam import *
from fastai.layers import MSELossFlat

mainfeature = torch.tensor(np.load('main_feat.npy')).float()
cosineloss = nn.CosineSimilarity(dim=2)

atfms = get_transforms(do_flip=True,max_rotate=360.,max_lighting=0.35,
                       xtra_tfms=zoom_crop(scale=(1,1.3),do_rand=True))

def freeze(md_ef):
    for idx,param in enumerate(md_ef.parameters()):
        if idx >= len(list(md_ef.parameters()))-2:
            param.requires_grad = True
        else:
            param.requires_grad = False

def metric(y_pred,y_true):
    yFeat = y_pred.clone()

    cosDis = cosineloss(yFeat.unsqueeze(1).repeat(1,5,1),
                             mainfeature.unsqueeze(0).repeat(len(y_true),1,1))

    weight = torch.ones(len(y_true),5).float()
    
    a = torch.arange(bs).long()
    
    weight[a,y_true.long()] = 0
    
    sim = (1. - cosDis[a,y_true.long()]).sum()
    
    dsim = (cosDis*weight).sum()

    return (sim,dsim)
    
def qappa_metric(y_pred, y):
    qappa_val = qappa(y_pred.max(1)[1].cpu(),y.cpu())
    return torch.tensor(qappa_val, device='cuda:0')

def qappa_met(y_pred,y):
    print(len(y_pred),len(y))
    qappa_val = qappa(torch.round(y_pred).cpu().long().view(-1),y.cpu())
    return torch.tensor(qappa_val, device='cuda:0')

def accuracy_metric(y_pred,y):
    return 100*(y_pred.max(1)[1] == y.long()).sum().float()/len(y)

def accuracy_met(y_pred,y):
    return 100*(y_pred.round().long().view(-1) == y.long()).sum().float()/len(y)

def cls_2_metric(y_pred,y):
    y_pred = y_pred.view(-1)
    idx = (y == 2)
    return accuracy_met(y_pred[idx],y[idx])

def other_cls(y_pred,y):
    y_pred = y_pred.view(-1)
    idx = (y != 2)
    return accuracy_met(y_pred[idx],y[idx])
    
from sklearn.metrics import cohen_kappa_score
def kappa_score(y_pred,y_true):
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()

    y_pred = y_pred.view(-1)
    y_pred[y_pred <= 0.5] = 0
    y_pred[(y_pred > 0.5) & (y_pred <= 1.5)] = 1
    y_pred[(y_pred > 1.5) & (y_pred <= 2.5)] = 2
    y_pred[(y_pred > 2.5) & (y_pred <= 3.5)] = 3
    y_pred[y_pred > 3.5] = 4

    return torch.tensor(cohen_kappa_score(y_true,y_pred,weights='quadratic'))