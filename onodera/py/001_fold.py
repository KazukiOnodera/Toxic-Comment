#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 14:17:37 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold

# Define KFold and random state
random_state = 4324455
n_splits = 5
kf = KFold(n_splits=n_splits, random_state=random_state)

train = pd.read_pickle('../data/{}2.p'.format('train'))


for fold_idx, (train_index, val_index) in enumerate(kf.split(train)):
    print('\nFold: {}'.format(fold_idx))
    # Train/validation dataset
    build_id, valid_id = train.iloc[train_index][['id']], train.iloc[val_index][['id']]
    
    build_id.to_pickle('../data/fold{}_build.p'.format(fold_idx))
    valid_id.to_pickle('../data/fold{}_valid.p'.format(fold_idx))

