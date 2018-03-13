#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:48:19 2018

@author: kazuki.onodera
"""

import numpy as np
import pandas as pd

# =============================================================================
# from ikki
# =============================================================================
files = ['../../ikki/output/fasttext_conv1_cv_oof_0.974374_0.002278.csv',
         '../../ikki/output/fasttext_conv1_cv_test.csv',
         '../../ikki/output/fasttext_correct_toxic_cv_oof_0.988551_0.000352.csv',
         '../../ikki/output/fasttext_correct_toxic_cv_test.csv',
         '../../ikki/output/fasttext_features_cv_oof_0.984296_0.000462.csv',
         '../../ikki/output/fasttext_features_cv_test.csv',
         '../../ikki/output/fasttext_lstm10_cudnn_cv_oof_0.989221_0.000313.csv',
         '../../ikki/output/fasttext_lstm10_cudnn_cv_test.csv',
         '../../ikki/output/fasttext_lstm11_cudnn_cv_oof_0.989831_0.000165.csv',
         '../../ikki/output/fasttext_lstm11_cudnn_cv_test.csv',
         '../../ikki/output/fasttext_lstm11_cudnn_prep_repeat_cv_oof_0.989970_0.000527.csv',
         '../../ikki/output/fasttext_lstm11_cudnn_prep_repeat_cv_test.csv',
         '../../ikki/output/fasttext_lstm12_cudnn_cv_oof_0.988968_0.000369.csv',
         '../../ikki/output/fasttext_lstm12_cudnn_cv_test.csv',
         '../../ikki/output/fasttext_lstm13_cudnn_cv_oof_0.978980_0.000754.csv',
         '../../ikki/output/fasttext_lstm13_cudnn_cv_test.csv',
         '../../ikki/output/fasttext_lstm16_cudnn_prep_repeat_cv_oof_0.989764_0.000310.csv',
         '../../ikki/output/fasttext_lstm16_cudnn_prep_repeat_cv_test.csv',
         '../../ikki/output/fasttext_lstm17_cudnn_prep_repeat_cv_oof_0.989660_0.000302.csv',
         '../../ikki/output/fasttext_lstm17_cudnn_prep_repeat_cv_test.csv',
         '../../ikki/output/fasttext_lstm1_cv_oof_0.988598_0.000409.csv',
         '../../ikki/output/fasttext_lstm1_cv_test.csv',
         '../../ikki/output/fasttext_lstm2_cv_oof_0.988600_0.000528.csv',
         '../../ikki/output/fasttext_lstm2_cv_test.csv',
         '../../ikki/output/fasttext_lstm3_cv_oof_0.983589_0.001340.csv',
         '../../ikki/output/fasttext_lstm3_cv_test.csv',
         '../../ikki/output/fasttext_lstm4_cudnn_cv_oof_0.987665_0.000706.csv',
         '../../ikki/output/fasttext_lstm4_cudnn_cv_test.csv',
         '../../ikki/output/fasttext_lstm5_cudnn_cv_oof_0.987135_0.000816.csv',
         '../../ikki/output/fasttext_lstm5_cudnn_cv_test.csv',
         '../../ikki/output/fasttext_lstm6_cudnn_cv_oof_0.987873_0.000603.csv',
         '../../ikki/output/fasttext_lstm6_cudnn_cv_test.csv',
         '../../ikki/output/fasttext_lstm7_cudnn_cv_oof_0.988155_0.000630.csv',
         '../../ikki/output/fasttext_lstm7_cudnn_cv_test.csv',
         '../../ikki/output/fasttext_lstm8_cudnn_cv_oof_0.983838_0.002173.csv',
         '../../ikki/output/fasttext_lstm8_cudnn_cv_test.csv',
         '../../ikki/output/fasttext_lstm9_cudnn_cv_oof_0.983592_0.000722.csv',
         '../../ikki/output/fasttext_lstm9_cudnn_cv_test.csv',
         '../../ikki/output/test_fasttext2_cv_oof_0.988522_0.000522.csv',
         '../../ikki/output/test_fasttext2_cv_test.csv']

cv_files = [f for f in files if 'cv_test' not in f]
test_files = [f for f in files if 'cv_test'  in f]

train = pd.concat([ pd.read_csv(f).set_index('id').add_suffix('_ikki-{}'.format(i)) for i,f in enumerate(cv_files) ], axis=1)
test  = pd.concat([ pd.read_csv(f).set_index('id').add_suffix('_ikki-{}'.format(i)) for i,f in enumerate(test_files) ], axis=1)

# =============================================================================
# label
# =============================================================================

labels = pd.read_pickle('../data/train2.p').drop('comment_text', axis=1).set_index('id')
label_col = labels.columns
#train = pd.merge(train, labels, on='id', how='left')
train = pd.concat([train, labels], axis=1)

X_train = train.drop(label_col, axis=1)
y_train = labels


# =============================================================================
# xgb
# =============================================================================
import sys
sys.path.append('/home/kazuki_onodera/Python')
import xgbextension as ex
import xgboost as xgb


param = {'colsample_bylebel': 0.2,
         'subsample': 0.3,
         'eta': 0.001,
         'eval_metric': 'auc',
         'max_depth': 3,
         'objective': 'binary:logistic',
         'silent': 1,
         'tree_method': 'hist',
         'seed':71}

models = []
for label in ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']:
    print(label)
    dtrain = xgb.DMatrix(X_train, y_train[label])
    model = xgb.train(param, dtrain, 12000)
    models.append(model)


dtest  = xgb.DMatrix(test)
y_preds = []
for model in models:
    y_pred = model.predict(dtest)
    y_preds.append(y_pred)


sub = test.reset_index()[['id']]
sub = pd.concat([sub, pd.DataFrame(y_preds).T], axis=1)
sub.columns = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

sub.to_csv('../output/313-2_onodera.csv.gz', index=False, compression='gzip')










