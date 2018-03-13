
# coding: utf-8

# In[17]:


import pandas as pd
from multiprocessing import Pool
total_proc = 64
#import sys
#sys.path.append('/home/kazuki_onodera/Python')
#import xgbextension as ex


# In[2]:


train = pd.read_pickle('../data/train1.p')
train_freq = pd.read_pickle('../data/train1_freq.p')


# # freq

# In[5]:


train.comment_text = train.comment_text.map(lambda x: x.split())


# In[23]:


label_col = train.columns[2:].tolist()
label_col


# In[13]:


def multi(word):
    df = train[train.comment_text.map(lambda x: word in x)]
    comment_freq = df.shape[0]
    return pd.Series(comment_freq).append(df[label_col].sum()).values


# In[14]:


multi('fuck')


# In[37]:


X_train = pd.DataFrame()
onehot_words = train_freq.word.tolist()


# In[38]:


pool = Pool(total_proc)
callback = pool.map(multi, onehot_words)
pool.close()


# In[40]:


df = pd.DataFrame(callback, index=onehot_words, columns=['comment_freq']+label_col)


# In[48]:


df = pd.concat([train_freq, df.reset_index(drop=1)], axis=1)


# In[50]:


df.to_pickle('../data/stem_vs_label.p')


# In[53]:


ls -lh ../data

