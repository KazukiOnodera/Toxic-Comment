#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 11 15:33:44 2018

@author: Kazuki
"""

#from gensim.models import Doc2Vec
from gensim.models import KeyedVectors
#from gensim.models import Word2Vec
w2v = KeyedVectors.load_word2vec_format('~/nlp_source/GoogleNews-vectors-negative300.bin.gz', binary=True)
import numpy as np
import pandas as pd
from tqdm import tqdm

np.random.seed(71)

# =============================================================================
# def
# =============================================================================
def valid_words(model, words):
    return [w for w in words if w in model]

def sen2vec(model, words, size=300):
    try:
        words = valid_words(model, words)
        if len(words)==0:
            raise
        vec   = [model[w] for w in words]
        return [np.array(vec).sum(axis=0)]
    except:
        return [np.zeros(size)]

def main(name, size=5):
    df = pd.read_pickle('../data/{}2.p'.format(name))
    df.comment_text = df.comment_text.map(lambda x: x.split())
    df['comment_length'] = df.comment_text.map(len)
    
    for j in range(10):
        # if size=5 then 4 -> 0, 5 -> 0, 6 -> 0or1
        df['start'] = df['comment_length'].map(lambda x: np.random.randint( max(x-size+1, 1) ))
        
        df['vec'] = df.apply(lambda x: sen2vec(w2v, x['comment_text'][x['start']:x['start']+size]), axis=1)
        for i in tqdm(range(300)):
            df['vec_{}'.format(i)] = df.vec.map(lambda x: x[0][i])
        
        col = ['id'] + [c for c in df.columns if 'vec_' in c]
        col = [c for c in df.columns if c not in col]
        df.drop(col, axis=1).to_pickle('../data/103_{}_{}.p'.format(name, j))
    
# =============================================================================
# main
# =============================================================================

main('train')
main('test')




