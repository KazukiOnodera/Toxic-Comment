import os
import re
import math
import csv
import itertools
from collections import Counter
import numpy as np
import pandas as pd
from fastText import load_model
import unidecode
import logging

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout, Activation, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, TimeDistributed, Permute, Reshape, Lambda, RepeatVector, Multiply, Concatenate

from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


window_length = 250 # The amount of words we look at per example. Experiment with this.

"""
Data loading
"""
print("Loading train/test data")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# target classes
classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
