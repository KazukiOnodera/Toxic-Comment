"""
特徴量作成?

参考
https://www.kaggle.com/mschumacher/using-fasttext-models-for-robust-embeddings/notebook

TODO: module化する
1. raw data -> preprocessing data
2. preprocessing data -> cross-validation -> prediction of out-of-fold and testset
3. stacking...
"""
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
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, TimeDistributed, Permute, Reshape, Lambda, RepeatVector, Multiply, Concatenate, SpatialDropout1D
from keras.layers import CuDNNGRU, CuDNNLSTM, Conv1D

from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.optimizers import SGD, RMSprop, Adam, Adamax
from keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


window_length = 350 # The amount of words we look at per example. Experiment with this.

"""
Data loading
"""
print("Loading train/test data")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# target classes
classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

"""
Feature engineering
"""
# count系
train_feat = pd.read_pickle('../data/102_train.p')
test_feat = pd.read_pickle('../data/102_test.p')


train = train.merge(train_feat, how='left', on='id')
test = test.merge(test_feat, how='left', on='id')

# Repeated sentencesを削除したテキスト
train_rm_repeat = pd.read_pickle('../data/train_rm_repeat.p')
test_rm_repeat = pd.read_pickle('../data/test_rm_repeat.p')

train_rm_repeat.columns = ['id', 'comment_text_rm_repeat']
test_rm_repeat.columns = ['id', 'comment_text_rm_repeat']

train_rm_repeat['comment_len_rm_repeat'] = train_rm_repeat.comment_text_rm_repeat.apply(len)
test_rm_repeat['comment_len_rm_repeat'] = test_rm_repeat.comment_text_rm_repeat.apply(len)

train = train.merge(train_rm_repeat, how='left', on='id')
test = test.merge(test_rm_repeat, how='left', on='id')

train['diff_comment_len_repeat'] = train.comment_len - train.comment_len_rm_repeat
test['diff_comment_len_repeat'] = test.comment_len - test.comment_len_rm_repeat

# Delete original text
del train['comment_text'], test['comment_text']

train.columns = train.columns.str.replace('comment_text_rm_repeat', 'comment_text')
test.columns = test.columns.str.replace('comment_text_rm_repeat', 'comment_text')

# normalization for neural network inputs
eng_feat_cols = ['comment_len', 'word_cnt', 'word_cnt_unq', 'word_max_len', 'comment_len_rm_repeat', 'diff_comment_len_repeat']

train[eng_feat_cols] = train[eng_feat_cols].applymap(lambda x: np.log10(x + 1) if x >= 0 else -np.log10(np.abs(x) + 1))
test[eng_feat_cols] = test[eng_feat_cols].applymap(lambda x: np.log10(x + 1) if x >= 0 else -np.log10(np.abs(x) + 1))



"""
Preprocessing functions
"""
# Remove accent marks
def remove_accent_before_tokens(sentences):
    res = unidecode.unidecode(sentences)
    return(res)

# Frequent Emphasized word


# Replace toxic words
def make_asterisk_toxic_word(toxic_word):
    split_word = list(toxic_word)
    product_set = [[split_char, '*'] if split_char != ' ' else [split_char] for split_char in split_word]
    asterisk_word = list(itertools.product(*product_set))
    asterisk_word_list = list(map(lambda x: ''.join(x), asterisk_word))
    # remove original word
    asterisk_word_list.remove(toxic_word)
    # remove all asterisk word
    asterisk_word_list = [word for word in asterisk_word_list if len(Counter(''.join(word.split())).values()) != 1]
    return asterisk_word_list

swear_words = list(csv.reader(open('../external_data/swearWords.csv', 'r')))[0]
ast_word2word = {ast_word: word for word in swear_words for ast_word in make_asterisk_toxic_word(word)}

# Expanding contraction
CONTRACTION_MAP = {"ain't": "is not", "aren't": "are not","can't": "cannot",
                   "can't've": "cannot have", "'cause": "because", "could've": "could have",
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not",
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
                   "he'll've": "he he will have", "he's": "he is", "how'd": "how did",
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will",
                   "I'll've": "I will have","I'm": "I am", "I've": "I have",
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will",
                   "i'll've": "i will have","i'm": "i am", "i've": "i have",
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have",
                   "it'll": "it will", "it'll've": "it will have","it's": "it is",
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not",
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have",
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have",
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would",
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have",
                   "she's": "she is", "should've": "should have", "shouldn't": "should not",
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is",
                   "there'd": "there would", "there'd've": "there would have","there's": "there is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will",
                   "they'll've": "they will have", "they're": "they are", "they've": "they have",
                   "to've": "to have", "wasn't": "was not", "we'd": "we would",
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have",
                   "we're": "we are", "we've": "we have", "weren't": "were not",
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                   "what's": "what is", "what've": "what have", "when's": "when is",
                   "when've": "when have", "where'd": "where did", "where's": "where is",
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have",
                   "who's": "who is", "who've": "who have", "why's": "why is",
                   "why've": "why have", "will've": "will have", "won't": "will not",
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not",
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                   "you'll've": "you will have", "you're": "you are", "you've": "you have" }

def expand_contractions(sentence, contraction_mapping):

    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())
        expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction

    expanded_sentence = contractions_pattern.sub(expand_match, sentence)
    return expanded_sentence

def normalize(s):
    """
    Given a text, cleans and normalizes it. Feel free to add your own stuff.
    """
    s = s.lower()
    # Special preprocessing
    s = s.replace('ı', 'i')
    s = s.replace('sh1t', 'shit')
    s = s.replace('f uck', 'fuck')
    s = s.replace('f u c k', 'fuck')
    s = s.replace('blow job', 'blowjob')
    # s = s.replace('\*uc\*', 'fuck')
    s = s.replace('God damn', 'goddamn')
    s = s.replace('knob end', 'knobend')

    s = s.replace('f.u.c.k', ' fuck ')
    s = s.replace('s.h.i.t', ' shit ')
    s = s.replace('c.u.n.t', ' cunt ')
    s = s.replace('a.s.s', ' ass ')
    s = s.replace('b.i.t.c.h', ' bitch ')
    s = s.replace(' youfuck ', ' you fuck ')
    s = s.replace(' youi ', ' you i ')
    s = s.replace(' youvvi ', ' you i ')
    s = s.replace('arsehole', ' asshole ')
    s = s.replace('schäbig', ' tackily ')
    s = s.replace('u.s.a', ' usa ')
    s = s.replace('hahahahello', ' hahaha hello ')
    s = s.replace('shiti', ' shit i ')
    s = s.replace(' fuk ', ' fuck ')
    s = s.replace('cocksuckers', ' cocksucker ')
    s = s.replace('youuhaxx0r', ' you hacker ')
    s = s.replace('whtat', ' what ')
    s = s.replace('go0verment', ' government ')
    s = s.replace('fuck youf', ' fuck you ')
    s = s.replace('mother fucjer', ' mother fucker ')
    s = s.replace('fuck u ', ' fuck you ')
    s = s.replace('fu ck ing ', ' fucking ')
    s = s.replace('motherfukkin', ' motherfucking ')
    s = s.replace('niggaz', ' niggas ')
    s = s.replace('fucken', ' fucking ')
    s = s.replace('fuckin ', ' fucking ')
    s = s.replace('fukiin ', ' fucking ')
    s = s.replace('shiit ', ' shit ')
    s = s.replace('stupiidestt ', ' stupidest ')
    s = s.replace('fuk-u', ' fuck you ')
    s = s.replace('fuking ', ' fucking ')
    s = s.replace('fukin ', ' fucking ')
    s = s.replace('fcuk ', ' fuck ')
    s = s.replace('fukers', ' fuck ')
    s = s.replace('fukk', ' fuck ')
    s = s.replace('afukin ', ' a fucking ')
    s = s.replace('mudafukars ', ' motherfucker ')
    s = s.replace('consfuc', ' fuck ')
    s = s.replace('familfuk', ' fuck ')
    s = s.replace('motherfuker', ' motherfucker ')
    s = s.replace('krucafuks', ' kruca fuck ')
    s = s.replace('fuken', ' fucking ')
    s = s.replace('kukk', ' fuck ')
    s = s.replace('ffukkkk', ' fuck ')
    s = s.replace('ffukkkkkkkkkkkkkkkkkkkkkk', ' fuck ')
    s = s.replace('fukn', ' fucking ')
    s = s.replace('fukinggggg', ' fucking ')
    s = s.replace('fukinggggggggggggggggggggggggggggg', ' fucking ')
    s = s.replace('fukking', ' fucking ')
    s = s.replace('fukcing', ' fucking ')

    # Expand contractions
    s = expand_contractions(s, CONTRACTION_MAP)

    # Remove email address
    s = re.sub(r'[^@]+@[^@]+\.[^@]+', 'EMAIL', s)

    # Remove links
    s = re.sub("(f|ht)tp(s?)://\\S+", "LINK", s)
    s = re.sub("http\\S+", "LINK", s)
    s = re.sub("xml\\S+", "LINK", s)

    # Remove modified text: f u c k  y o u => fuck you
    # s = re.sub("(?<=\\b\\w)\\s(?=\\w\\b)", "", s)
    # Remeve shit text
    # s = re.sub("\\b(a|e)w+\\b", "AWWWW", s)
    # s = re.sub("\\b(y)a+\\b", "YAAAA", s)
    # s = re.sub("\\b(w)w+\\b", "WWWWW", s)

    # s = re.sub("\\b(b+)?((h+)((a|e|i|o|u)+)(h+)?){2,}\\b", "HAHEHI", s)
    # s = re.sub("\\b(b+)?(((a|e|i|o|u)+)(h+)((a|e|i|o|u)+)?){2,}\\b", "HAHEHI", s)
    # s = re.sub("\\b(m+)?(u+)?(b+)?(w+)?((a+)|(h+))+\\b", "HAHEHI", s)
    # s = re.sub("\\b((e+)(h+))+\\b", "HAHEHI", s)
    # s = re.sub("\\b((h+)(e+))+\\b", "HAHEHI", s)
    # s = re.sub("\\b((o+)(h+))+\\b", "HAHEHI", s)
    # s = re.sub("\\b((h+)(o+))+\\b", "HAHEHI", s)
    # s = re.sub("\\b((l+)(a+))+\\b", "LALALA", s)
    # s = re.sub("(w+)(o+)(h+)(o+)", "WOHOO", s)
    # s = re.sub("\\b(d?(u+)(n+)?(h+))\\b", "UUUHHH", s)
    # s = re.sub("\\b(a+)(r+)(g+)(h+)\\b", "ARGH", s)
    # s = re.sub("\\b(a+)(w+)(h+)\\b", "AAAWWHH", s)
    # s = re.sub("\\b(p+)(s+)(h+)\\b", "SHHHHH", s)
    # s = re.sub("\\b((s+)(e+)?(h+))+\\b", "SHHHHH", s)
    # s = re.sub("\\b(s+)(o+)\\b", "", s)
    # s = re.sub("\\b(h+)(m+)\\b", "HHMM", s)
    # s = re.sub("\\b((b+)(l+)(a+)(h+)?)+\\b", "BLABLA", s)
    # s = re.sub("\\b((y+)(e+)(a+)(h+)?)+\\b", "YEAH", s)
    # s = re.sub("\\b((z+)?(o+)(m+)(f+)?(g+))+\\b", "OMG", s)
    # s = re.sub("aa(a+)", "a", s)
    # s = re.sub("ee(e+)", "e", s)
    # s = re.sub("i(i+)", "i", s)
    # s = re.sub("oo(o+)", "o", s)
    # s = re.sub("uu(u+)", "u", s)
    # s = re.sub("\\b(u(u+))\\b", "u", s)
    # s = re.sub("y(y+)", "y", s)
    # s = re.sub("hh(h+)", "h", s)
    # s = re.sub("gg(g+)", "g", s)
    # s = re.sub("tt(t+)\\b", "t", s)
    # s = re.sub("(tt(t+))", "tt", s)
    # s = re.sub("mm(m+)", "m", s)
    # s = re.sub("ff(f+)", "f", s)
    # s = re.sub("cc(c+)", "c", s)
    # s = re.sub("\\b(kkk)\\b", "KKK", s)
    # s = re.sub("\\b(pkk)\\b", "PKK", s)
    # s = re.sub("kk(k+)", "kk", s)
    # s = re.sub("fukk", "fuck", s)
    # s = re.sub("k(k+)\\b", "k", s)
    # s = re.sub("f+u+c+k+\\b", "fuck", s)
    # s = re.sub("((a+)|(h+)){3,}", "HAHEHI", s)

    # Remove modified text: f u c k  y o u => fuck you
    # s = re.sub("(?<=\\b\\w)\\s(?=\\w\\b)", "", s)

    s = re.sub("((lol)(o?))+\\b", "LOL", s)
    s = re.sub("n ig ger", "nigger", s)
    s = re.sub("nig ger", "nigger", s)
    s = re.sub("s hit", "shit", s)
    s = re.sub("g ay", "gay", s)
    s = re.sub("f ag got", "faggot", s)
    s = re.sub("c ock", "cock", s)
    s = re.sub("cu nt", "cunt", s)
    s = re.sub("idi ot", "idiot", s)
    # s = re.sub("(?<=\\b(fu|su|di|co|li))\\s(?=(ck)\\b)", "", s)
    # #gsub("(?<=\\w(ck))\\s(?=(ing)\\b)", "", "fuck ing suck ing lick ing", perl = T)
    # s = re.sub("(?<=\\w(ck))\\s(?=(ing)\\b)", "", s)

    # Remove accent marks
    #s = remove_accent_before_tokens(s)
    # Replace ips
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    # Isolate punctuation
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,\#])', r' \1 ', s).replace('\1', '')
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n=:;",.\/—\-\(\)~\[\]\_\#])', ' ', s)
    # Remove special word
    s = re.sub(r'([☺☻♥♦♣♠•◘○♂♀♪♫☼►◄])', ' ', s)
    # Remove repeated (consecutive) words
    #TODO: 繋がっている単語はわけられない　'fuck fuck'=>'fuck', 'FUCKFUCK'=>'FUCKFUCK'
    s = re.sub(r'\b(\w+)( \1\b)+', r'\1', s)


    # Remove new lines
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    # s = s.replace('0', ' zero ')
    # s = s.replace('1', ' one ')
    # s = s.replace('2', ' two ')
    # s = s.replace('3', ' three ')
    # s = s.replace('4', ' four ')
    # s = s.replace('5', ' five ')
    # s = s.replace('6', ' six ')
    # s = s.replace('7', ' seven ')
    # s = s.replace('8', ' eight ')
    # s = s.replace('9', ' nine ')

    # Remove number
    s = re.sub(r'\b\d+(?:\.\d+)?\s+', ' ', s)
    return s

"""
Util functions for keras
"""
class IntervalEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.score_hitory = []

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred, average=None, sample_weight=None)
            score_mean, score_std = np.mean(score), np.std(score)
            self.score_hitory.append([score_mean, score_std])
            logging.info("interval evaluation - epoch: {:d} - score: {:.6f}+{:.6f}".format(epoch, score_mean, score_std))
            print("\ninterval evaluation - epoch: {:d} - score: {:.6f}+{:.6f}".format(epoch, score_mean, score_std))

"""
Function to generate training data
"""
def text_to_vector(text):
    """
    Given a string, normalizes it, then splits it into words and finally converts
    it to a sequence of word vectors.
    ---
    window_lengthよりも大きい文章のときスタート位置をランダムにする機能追加
    """
    #text = normalize(text)
    words = text.split()
    #TODO: remove stop words(https://www.kaggle.com/saxinou/nlp-01-preprocessing-data)
    # *を含む単語を置換する
    words = [ast_word2word.get(word) if ast_word2word.get(word) is not None else word for word in words]
    # 繰り視される文字をゆにーくにする+ fuuuuuuuuuck -> fuck
    words = [re.sub(r'([a-z]+?)\1+', r'\1', word) if len(word) > 30 else word for word in words]
    # 置換後は*を取り除く?
    #words = [word.replace('*', '') for word in words]
    words_num = len(words)
    if words_num <= window_length:
        window = words
    else:
        window_start_idx = np.random.choice(words_num - window_length + 1, 1)[0]
        window = words[window_start_idx:window_length]

    x = np.zeros((window_length, n_features))

    for i, word in enumerate(window):
        x[i, :] = ft_model.get_word_vector(word).astype('float32')

    return x

def df_to_data(df):
    """
    Convert a given dataframe to a dataset of inputs for the NN.
    """
    x = np.zeros((len(df), window_length, n_features), dtype='float32')

    for i, comment in enumerate(df['comment_text'].values):
        x[i, :] = text_to_vector(comment)

    return x

def data_generator(df, batch_size):
    """
    Given a raw dataframe, generates infinite batches of FastText vectors.
    """
    batch_i = 0 # Counter inside the current batch vector
    batch_x = None # The current batch's x data
    batch_y = None # The current batch's y data

    batch_x_feat = None # The current batch's x data

    while True: # Loop forever
        df = df.sample(frac=1) # Shuffle df each epoch

        for i, row in df.iterrows():
            comment = row['comment_text']

            if batch_x is None:
                batch_x = np.zeros((batch_size, window_length, n_features), dtype='float32')
                batch_x_feat = np.zeros((batch_size, len(eng_feat_cols)), dtype='float32')
                batch_y = np.zeros((batch_size, len(classes)), dtype='float32')

            batch_x[batch_i] = text_to_vector(comment)
            batch_x_feat[batch_i] = row[eng_feat_cols]
            batch_y[batch_i] = row[classes].values
            batch_i += 1

            if batch_i == batch_size:
                # Ready to yield the batch
                yield [batch_x, batch_x_feat], batch_y
                batch_x = None
                batch_x_feat = None
                batch_y = None
                batch_i = 0

def data_generator_for_test(df, batch_size):
    """
    Given a raw dataframe, generates infinite batches of FastText vectors.
    """
    df_length = len(df)
    batch_i = 0 # Counter inside the current batch vector
    all_data_i = 0
    batch_x = None # The current batch's x data
    batch_x_feat = None # The current batch's x data

    for i, row in df.iterrows():
        comment = row['comment_text']

        if batch_x is None:
            batch_x = np.zeros((batch_size, window_length, n_features), dtype='float32')
            batch_x_feat = np.zeros((batch_size, len(eng_feat_cols)), dtype='float32')

        batch_x[batch_i] = text_to_vector(comment)
        batch_x_feat[batch_i] = row[eng_feat_cols]
        batch_i += 1
        all_data_i += 1

        if batch_i == batch_size:
            # Ready to yield the batch
            yield [batch_x, batch_x_feat]
            batch_x = None
            batch_x_feat = None
            batch_i = 0
        elif all_data_i == df_length:
            # Ready to yield the last batch
            yield [batch_x[:batch_i], batch_x_feat[:batch_i]]
            batch_x = None
            batch_x_feat = None
            batch_i = 0

"""
Preprocessings
"""
# Replace nan value
#train['comment_text'] = train['comment_text'].fillna('_empty_')
#test['comment_text'] = test['comment_text'].fillna('_empty_')

# Replace miscellaneous characters
train['comment_text'] = train['comment_text'].str.replace('ı', 'i')
test['comment_text'] = test['comment_text'].str.replace('ı', 'i')

# 両端に*がある単語を取得
# 存在するものは除く
tmp = pd.concat([train, test]).comment_text.apply(normalize)
tmp2 = tmp.str.split().apply(lambda x: [w for w in x if (w[0] == w[-1] == '*') or (len(w) >= 5 and w[0] == w[1] == w[-2] == w[-1] == '*')])
tmp3 = tmp2[tmp2.apply(len)!=0]
ast_between_word = pd.Series(np.concatenate(tmp3.tolist())).value_counts()
delete_key_words = [word for word in ast_between_word.index if ast_word2word.get(word) is not None]
for del_key in delete_key_words:
    del ast_word2word[del_key]

# Normalize comment_text (IMPLEMENTED IN GENERATOR)
#train['comment_text'] = train['comment_text'].apply(normalize)
#test['comment_text'] = test['comment_text'].apply(normalize)

# Split comment_text (IMPLEMENTED IN GENERATOR)
#train['comment_text'] = train['comment_text'].str.split()
#test['comment_text'] = test['comment_text'].str.split()

"""
Loading FT model
"""
print('Loading FT model')
ft_model = load_model('../external_data/pretrained/fasttext/wiki/wiki.en.bin')
# Embedding dimension
n_features = ft_model.get_dimension()


"""
Define models
"""

def build_lstm_stack_model(logdir='attention'):
    # Bidirectional-LSTM
    inp = Input(shape=(window_length, 300))
    inp_dr = SpatialDropout1D(0.05)(inp)
    l_lstm = Bidirectional(CuDNNGRU(256, return_sequences=True))(inp_dr)
    l_lstm = Dropout(0.05)(l_lstm)
    x_gmp = GlobalMaxPool1D()(l_lstm)
    x_gap = GlobalAveragePooling1D()(l_lstm)
    x_gmp_gap = Concatenate()([x_gmp, x_gap])
    x_gmp_gap = Dropout(0.1)(x_gmp_gap)

    inp_feat = Input(shape=(len(eng_feat_cols),))

    x = Concatenate()([x_gmp_gap, inp_feat])

    x = Dense(256, activation="elu")(x)
    x = Dropout(0.25)(x)
    x = Dense(6, activation="sigmoid")(x)

    model = Model(inputs=[inp, inp_feat], outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=5e-4, amsgrad=True), metrics=['accuracy'])

    return model

"""
Normalize comment before training
"""
train['comment_text'] = train['comment_text'].apply(normalize)
test['comment_text'] = test['comment_text'].apply(normalize)

"""
Training and evaluating with cross-validation
"""
# Filename to save
saving_filename = 'fasttext_lstm11_cudnn_prep_repeat_cv'

# Define KFold and random state
random_state = 4324455
n_splits = 5
kf = KFold(n_splits=n_splits, random_state=random_state)

# Make test dataset to fixed array
#print('Converting test dataframe to array')
#x_test = df_to_data(test)

# Cross validation
# Prediction of out-of-fold data
pred_oof = pd.concat([train['id'], pd.DataFrame(np.zeros((train.shape[0], len(classes))), columns=classes)], axis=1)
auc_pred_oof = []
# Prediction of test dataset
pred_test = pd.read_csv('../input/sample_submission.csv')
pred_test[classes] = 0.0  #> initialize


for fold_idx, (train_index, val_index) in enumerate(kf.split(train)):
    print('\nFold: {}'.format(fold_idx))
    # Train/validation dataset
    x_train, x_val = train.iloc[train_index], train.iloc[val_index]
    y_train, y_val = train[classes].iloc[train_index], train[classes].iloc[val_index]
    y_val = y_val.values
    # Convert validation set to fixed array
    print('Converting validation dataframe to array')
    #x_val = df_to_data(x_val)
    #validation_generator = data_generator_for_test(x_val, 128)
    #y_val = y_val.values

    # Build model
    print('Building model')
    model = build_lstm_stack_model()

    # Parameters
    training_epochs = 5
    batch_size = 96
    training_steps_per_epoch = math.ceil(len(x_train) / batch_size)
    training_generator = data_generator(x_train, batch_size)

    # Callbacks
    ival = IntervalEvaluation(validation_data=(x_val, y_val), interval=1)

    # Training
    print('Training model')
    for epoch in range(training_epochs):
        callback_history = model.fit_generator(
                                training_generator,
                                steps_per_epoch=training_steps_per_epoch,
                                max_queue_size=50,
                                workers=4,
                                epochs=1,
                                )

        # Predict at validation dataset
        print('Validating')
        validation_batch_size = 128
        validation_steps = math.ceil(len(x_val) / validation_batch_size)
        validation_generator = data_generator_for_test(x_val, validation_batch_size)
        y_val_pred = model.predict_generator(validation_generator, steps=validation_steps, verbose=1)
        score = roc_auc_score(y_val, y_val_pred, average=None, sample_weight=None)
        score_mean, score_std = np.mean(score), np.std(score)
        print("\ninterval evaluation - epoch: {:d} - score: {:.6f}+{:.6f}".format(epoch + 1, score_mean, score_std))


    # Predict at validation dataset
    print('Validating')
    validation_batch_size = 128
    validation_steps = math.ceil(len(x_val) / validation_batch_size)
    pred_val_num = 10
    y_val_preds = np.array([])
    for i in range(pred_val_num):
        validation_generator = data_generator_for_test(x_val, validation_batch_size)
        if i == 0:
            y_val_pred = model.predict_generator(validation_generator, steps=validation_steps, verbose=1)
            assert len(y_val_pred) == len(x_val)
            y_val_preds = np.expand_dims(y_val_pred, 2)
        else:
            y_val_pred = model.predict_generator(validation_generator, steps=validation_steps, verbose=1)
            y_val_pred = np.expand_dims(y_val_pred, 2)
            y_val_preds = np.concatenate([y_val_preds, y_val_pred], axis=2)
    y_val_preds_max = y_val_preds.max(2)

    # Asign results to dataframe
    pred_oof.loc[val_index, classes] = y_val_preds_max

    # Evaluate validation results
    auc_val = roc_auc_score(y_val, y_val_preds_max, average=None, sample_weight=None)
    auc_val_mean = np.mean(auc_val)
    auc_val_std = np.std(auc_val)
    auc_pred_oof.append([auc_val_mean, auc_val_std])
    print('Averaged AUC of validation: {}+{}'.format(auc_val_mean, auc_val_std))


    # Predict test dataset several times at random
    print('Testing')
    testing_batch_size = 128
    testing_steps = math.ceil(len(test) / testing_batch_size)
    pred_test_num = 10
    y_test = np.array([])
    for i in range(pred_test_num):
        testing_generator = data_generator_for_test(test, testing_batch_size)
        if i == 0:
            y_test_pred = model.predict_generator(testing_generator, steps=testing_steps, verbose=1)
            assert len(y_test_pred) == len(test)
            y_test = np.expand_dims(y_test_pred, 2)
        else:
            y_test_pred = model.predict_generator(testing_generator, steps=testing_steps, verbose=1)
            y_test_pred = np.expand_dims(y_test_pred, 2)
            y_test = np.concatenate([y_test, y_test_pred], axis=2)
    y_test = y_test.max(2)

    # Asign results to dataframe (divide followed by sum)
    pred_test[classes] += y_test / n_splits

    # Save model every fold
    #from keras.models import load_model
    model.save('../model/model_{}_{}.h5'.format(saving_filename, fold_idx))  # creates a HDF5 file 'my_model.h5'

# AUC of cross-validation
auc_pred_oof_fold = [auc_mean for auc_mean, auc_std in auc_pred_oof]
auc_pred_oof_mean = np.mean(auc_pred_oof_fold)
auc_pred_oof_std = np.std(auc_pred_oof_fold)

# Save result of cross-validation
pred_oof.to_csv('../output/{}_oof_{:.06f}_{:.06f}.csv'.format(saving_filename, auc_pred_oof_mean, auc_pred_oof_std), index=False)
pred_test.to_csv('../output/{}_test.csv'.format(saving_filename), index=False)


"""
sample_submission = pd.read_csv('../input/sample_submission.csv')
test_fasttext1 = pd.read_csv('../output/test_fasttext_val0.9893_0.0037.csv')
test_fasttext2 = pd.read_csv('../output/test_fasttext_val0.990713_0.003469.csv')
sample_submission[classes] = (test_fasttext1[classes] + test_fasttext2[classes]) / 2
sample_submission.to_csv('../output/avg_test_fasttext1_test_fasttext2.csv', index=False)
# fasttext1: 0.9831
# fasttext2: 0.9841
# fasttext1+2: 0.9847


fasttext2_cv = pd.read_csv('../output/test_fasttext2_cv_oof_0.988522_0.000522.csv')
fasttext_correct_toxic_cv = pd.read_csv('../output/fasttext_correct_toxic_cv_oof_0.988551_0.000352.csv')
fasttext_lstm2_cv = pd.read_csv('../output/fasttext_lstm2_cv_oof_0.988600_0.000528.csv')
fasttext_lstm1_cv = pd.read_csv('../output/fasttext_lstm1_cv_oof_0.988598_0.000409.csv')

cv_emsenble = (fasttext2_cv[classes] + \
               fasttext_correct_toxic_cv[classes] + \
               fasttext_lstm2_cv[classes] + \
               fasttext_lstm1_cv[classes]) / 4

auc_val = roc_auc_score(train[classes].values, cv_emsenble, average=None, sample_weight=None)


sample_submission = pd.read_csv('../input/sample_submission.csv')
avg_test_fasttext1_test_fasttext2_test_fasttext5 = pd.read_csv('../output/avg_test_fasttext1_test_fasttext2_test_fasttext5.csv')
fasttext2_cv = pd.read_csv('../output/test_fasttext2_cv_test.csv')
fasttext_correct_toxic_cv = pd.read_csv('../output/fasttext_correct_toxic_cv_test.csv')
fasttext_lstm2_cv = pd.read_csv('../output/fasttext_lstm2_cv_test.csv')
fasttext_lstm1_cv = pd.read_csv('../output/fasttext_lstm1_cv_test.csv')


sample_submission[classes] = (avg_test_fasttext1_test_fasttext2_test_fasttext5[classes] + \
                fasttext2_cv[classes] + \
               fasttext_correct_toxic_cv[classes] + \
               fasttext_lstm2_cv[classes] + \
               fasttext_lstm1_cv[classes]) / 5
sample_submission.to_csv('../output/fasttext2_cv_fasttext_correct_toxic_cv_lstm1_2_features.csv', index=False)
# fasttext2_cv_avg9857: 0.9860
# fasttext2_cv_avg9857_fasttext_correct_toxic_cv: 0.9862
# fasttext2_cv_fasttext_correct_toxic_cv_lstm1_2: 0.9861

"""