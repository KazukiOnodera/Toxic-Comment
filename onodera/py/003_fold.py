#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 22:07:01 2018

@author: Kazuki
"""

import pandas as pd
import numpy as np

fold = 10

train = pd.read_pickle('../data/train2.p')[['id']]
test  = pd.read_pickle('../data/test2.p')[['id']]



train['fold'] = train.index%10
test['fold'] = test.index%10



train.to_pickle('../data/train_fold.p')
test.to_pickle('../data/test_fold.p')

