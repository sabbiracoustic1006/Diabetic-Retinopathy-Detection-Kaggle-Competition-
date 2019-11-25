#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 16:59:17 2019

@author: ratul
"""

import os
import pandas as pd

def get_df():
    train_dir = 'train_images_256/'
    df = pd.read_csv('train.csv')
    df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))
    df = df.drop(columns=['id_code'])
    df = df.reset_index(drop=True) #shuffle dataframe
    test_df = pd.read_csv('sample_submission.csv')
    return df, test_df

def get_df_another():
    train_dir = 'train_images_previous/'
    df = pd.read_csv('trainLabels.csv')
    df['path'] = df['image'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))
    df['diagnosis'] = df['level'].map(lambda x: x)
    df = df.drop(columns=['image','level'])
    df = df.reset_index(drop=True) #shuffle dataframe
    test_df = pd.read_csv('sample_submission.csv')
    return df, test_df

def one_df_not_in_another_final(df,df_another):
    common = df.merge(df_another,on=['diagnosis','path'])
    dd__ = df[(~df.path.isin(common.path))]
    return dd__

df1, test_df = get_df()
df2, _ = get_df_another()
df3 = pd.read_csv('finalTest15.csv')

df_pseudo = pd.read_csv('pseudolabel.csv')

def process_dfs(*dfs):
    paths = []; diagnosis = [];
    for df in dfs:
        paths += list(df['path'].values)
        diagnosis += list(df['diagnosis'].values)
    return pd.DataFrame({'path':paths,'diagnosis':diagnosis}).sample(frac=1,random_state=1)