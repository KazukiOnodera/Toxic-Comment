# Modifying Tilii's original scrips https://www.kaggle.com/tilii7/tuned-logreg-oof-files
# With https://www.kaggle.com/tunguz/cnn-glove300-3-oof-4-epochs
# The output oof and prediction files could potentially be used for ensembling
# This version uses 5 consistent folds.

import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score, KFold
from scipy.sparse import hstack
from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score
from datetime import datetime

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod(
            (datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' %
              (thour, tmin, round(tsec, 2)))

# Data processing was done as in Bojan's fork of the original script:
# https://www.kaggle.com/tunguz/logistic-regression-with-words-and-char-n-grams

class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

traintime = timer(None)
train_time = timer(None)
train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')

#train = train.head(3000)
#test = test.head(3000)

tr_ids = train[['id']]
train[class_names] = train[class_names].astype(np.int8)
target = train[class_names]

print(' Cleaning ...')
# PREPROCESSING PART
repl = {
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " frown ",
    ":(": " frown ",
    ":s": " frown ",
    ":-s": " frown ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "m": "am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll" : "i will",
    "its" : "it is",
    "it's" : "it is",
    "'s" : " is",
    "that's" : "that is",
    "weren't" : "were not",
}

repl = {
    "yay!": " good ",
    "yay": " good ",
    "yaay": " good ",
    "yaaay": " good ",
    "yaaaay": " good ",
    "yaaaaay": " good ",
    ":/": " bad ",
    ":&gt;": " sad ",
    ":')": " sad ",
    ":-(": " frown ",
    ":(": " frown ",
    ":s": " frown ",
    ":-s": " frown ",
    "&lt;3": " heart ",
    ":d": " smile ",
    ":p": " smile ",
    ":dd": " smile ",
    "8)": " smile ",
    ":-)": " smile ",
    ":)": " smile ",
    ";)": " smile ",
    "(-:": " smile ",
    "(:": " smile ",
    ":/": " worry ",
    ":&gt;": " angry ",
    ":')": " sad ",
    ":-(": " sad ",
    ":(": " sad ",
    ":s": " sad ",
    ":-s": " sad ",
    r"\br\b": "are",
    r"\bu\b": "you",
    r"\bhaha\b": "ha",
    r"\bhahaha\b": "ha",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bhasn't\b": "has not",
    r"\bhaven't\b": "have not",
    r"\bhadn't\b": "had not",
    r"\bwon't\b": "will not",
    r"\bwouldn't\b": "would not",
    r"\bcan't\b": "can not",
    r"\bcannot\b": "can not",
    r"\bi'm\b": "i am",
    "m": "am",
    "r": "are",
    "u": "you",
    "haha": "ha",
    "hahaha": "ha",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "hasn't": "has not",
    "haven't": "have not",
    "hadn't": "had not",
    "won't": "will not",
    "wouldn't": "would not",
    "can't": "can not",
    "cannot": "can not",
    "i'm": "i am",
    "m": "am",
    "i'll" : "i will",
    "its" : "it is",
    "it's" : "it is",
    "'s" : " is",
    "that's" : "that is",
    "weren't" : "were not",
}

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


repl.update(CONTRACTION_MAP)

keys = [i for i in repl.keys()]

def normalize(s):
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
    s = re.sub("n ig ger", "nigger", s)
    s = re.sub("nig ger", "nigger", s)
    s = re.sub("s hit", "shit", s)
    s = re.sub("g ay", "gay", s)
    s = re.sub("f ag got", "faggot", s)
    s = re.sub("c ock", "cock", s)
    s = re.sub("cu nt", "cunt", s)
    s = re.sub("idi ot", "idiot", s)
    return s

train['comment_text'] = train['comment_text'].apply(normalize)
test['comment_text'] = test['comment_text'].apply(normalize)

new_train_data = []
new_test_data = []
ltr = train["comment_text"].tolist()
lte = test["comment_text"].tolist()
for i in ltr:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_train_data.append(xx)
for i in lte:
    arr = str(i).split()
    xx = ""
    for j in arr:
        j = str(j).lower()
        if j[:4] == 'http' or j[:3] == 'www':
            continue
        if j in keys:
            # print("inn")
            j = repl[j]
        xx += j + " "
    new_test_data.append(xx)
train["new_comment_text"] = new_train_data
test["new_comment_text"] = new_test_data

trate = train["new_comment_text"].tolist()
tete = test["new_comment_text"].tolist()
for i, c in enumerate(trate):
    trate[i] = re.sub('[^a-zA-Z ?!]+', '', str(trate[i]).lower())
for i, c in enumerate(tete):
    tete[i] = re.sub('[^a-zA-Z ?!]+', '', tete[i])
train["comment_text"] = trate
test["comment_text"] = tete
del trate, tete
train.drop(["new_comment_text"], axis=1, inplace=True)
test.drop(["new_comment_text"], axis=1, inplace=True)

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])
timer(train_time)

train_time = timer(None)
print(' Part 1/2 of vectorizing ...')
word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=50000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)
timer(train_time)

train_time = timer(None)
print(' Part 2/2 of vectorizing ...')
char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)
timer(train_time)

train_features = hstack([train_char_features, train_word_features]).tocsr()
test_features = hstack([test_char_features, test_word_features]).tocsr()
timer(traintime)

all_parameters = {
                  'C'             : [1.048113, 0.1930, 0.596362, 0.25595, 0.449843, 0.25595],
                  'tol'           : [0.1, 0.1, 0.046416, 0.0215443, 0.1, 0.01],
                  'solver'        : ['lbfgs', 'newton-cg', 'lbfgs', 'newton-cg', 'newton-cg', 'lbfgs'],
                  'fit_intercept' : [True, True, True, True, True, True],
                  'penalty'       : ['l2', 'l2', 'l2', 'l2', 'l2', 'l2'],
                  'class_weight'  : [None, 'balanced', 'balanced', 'balanced', 'balanced', 'balanced'],
                 }

folds = 5
scores = []
scores_classes = np.zeros((len(class_names), folds))

submission = pd.DataFrame.from_dict({'id': test['id']})
submission_oof = train[['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
#skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
kf = KFold(n_splits=folds, random_state=4324455)

idpred = tr_ids

traintime = timer(None)
for j, (class_name) in enumerate(class_names):
#    train_target = train[class_name]

    classifier = LogisticRegression(
        C=all_parameters['C'][j],
        max_iter=250,
        tol=all_parameters['tol'][j],
        solver=all_parameters['solver'][j],
        fit_intercept=all_parameters['fit_intercept'][j],
        penalty=all_parameters['penalty'][j],
        dual=False,
        class_weight=all_parameters['class_weight'][j],
        n_jobs=6,
        verbose=1)

    avreal = target[class_name]
    lr_cv_sum = 0
    lr_pred = []
    lr_fpred = []
    lr_avpred = np.zeros(train.shape[0])

    train_time = timer(None)
    for i, (train_index, val_index) in enumerate(kf.split(train_features)):
        X_train, X_val = train_features[train_index], train_features[val_index]
        y_train, y_val = target.loc[train_index], target.loc[val_index]

        classifier.fit(X_train, y_train[class_name])
        scores_val = classifier.predict_proba(X_val)[:, 1]
        lr_avpred[val_index] = scores_val
        lr_y_pred = classifier.predict_proba(test_features)[:, 1]
        scores_classes[j][i] = roc_auc_score(y_val[class_name], scores_val)
        print('\n Fold %02d class %s AUC: %.6f' % ((i+1), class_name, scores_classes[j][i]))

        if i > 0:
            lr_fpred = lr_pred + lr_y_pred
        else:
            lr_fpred = lr_y_pred

        lr_pred = lr_fpred

    lr_cv_score = (lr_cv_sum / folds)
    lr_oof_auc = roc_auc_score(avreal, lr_avpred)
    print('\n Average class %s AUC:\t%.6f' % (class_name, np.mean(scores_classes[j])))
    print(' Out-of-fold class %s AUC:\t%.6f' % (class_name, lr_oof_auc))
    timer(train_time)

    submission[class_name] = lr_pred / folds
    submission_oof[class_name] = lr_avpred

auc_oof_mean = np.mean(np.mean(scores_classes, 1))
auc_oof_std = np.mean(np.std(scores_classes, 1))

saving_filename = 'tuned_LR8_cv'

print('\n Overall AUC: %.6f+/-%.6f' % (auc_oof_mean, auc_oof_std))
submission.to_csv('../output/{}_test.csv'.format(saving_filename), index=False)
submission_oof.to_csv('../output/{}_oof_{:.06f}_{:.06f}.csv'.format(saving_filename, auc_oof_mean, auc_oof_std), index=False)

timer(traintime)
