
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np

import wordbatch
from wordbatch.extractors import WordBag, WordHash

from sklearn.feature_extraction.text import  TfidfVectorizer


# In[2]:


train = pd.read_pickle('../data/train1.p')
test = pd.read_pickle('../data/test1.p')


# In[3]:


train.sample(9)


# In[4]:


tv = TfidfVectorizer(max_features=50000, ngram_range=(1,3))


# In[5]:


train_tv = tv.fit_transform(train['comment_text'])


# In[9]:


test_tv = tv.transform(test['comment_text'])


# In[11]:


wb = wordbatch.WordBatch(extractor=(WordBag, {"hash_ngrams": 2, "hash_ngrams_weights": [1.5, 1.0],
                                                              "hash_size": 2 ** 29, "norm": None, "tf": 'binary',
                                                              "idf": None,
                                                              }), procs=8)
wb.dictionary_freeze= True


# In[12]:


train_wb = wb.fit_transform(train['comment_text'])
test_wb = wb.transform(test['comment_text'])


# In[13]:


train_wb


# In[21]:


np.savez('../data/101_train.npz', tv=train_tv, wb=train_wb)


# In[20]:


np.savez('../data/101_test.npz', tv=test_tv, wb=test_wb)

