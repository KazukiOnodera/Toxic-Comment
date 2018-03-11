#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 11:07:48 2018

@author: Kazuki
"""

import pandas as pd
stopwords = {'is', 'am', 'are', 'a', 'the'}


# =============================================================================
# def
# =============================================================================
def remove_stop(s):
    return [w for w in s.split() if w not in stopwords]

def after_you(s, size=3):
    """
    ex1:
        [you, jerk, you, deserve, yaoi, ohhhhh] -> [jerk], [deserve, yaoi, ohhhhh]
    ex2:
        [you, are, very, rude]  -> [are, very, rude]
    """
    if 'you' not in s :
        return
    
    sw = False
    cnt = 0
    ret = []
    for w in s:
        
        if sw == False and w in {'you'}:
            #print(w, 1) # for debug
            sw = True
            li = []
            continue
        
        if sw == True and w in {'you'}:
            #print(w, 2)
            cnt = 0
            ret.append(li)
            li = []
        elif cnt==size :
            #print(w, 3)
            cnt = 0
            ret.append(li)
            li = []
            sw = False
        elif sw:
            #print(w, 4)
            cnt +=1
            li.append(w)
            
    if len(li)!=0:
        ret.append(li)
        
    return ret


# =============================================================================
# main
# =============================================================================

train = pd.read_pickle('../data/train2.p')
test  = pd.read_pickle('../data/test2.p')

train.comment_text = train.comment_text.map(remove_stop)
train['after_you'] = train.comment_text.map(after_you)

test.comment_text = test.comment_text.map(remove_stop)
test['after_you'] = test.comment_text.map(after_you)


col = ['id', 'after_you']
train[col].to_pickle('../data/train_after_you.p')
test[col].to_pickle('../data/test_after_you.p')



