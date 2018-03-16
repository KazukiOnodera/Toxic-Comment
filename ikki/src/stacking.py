import os
from itertools import product

from tqdm import tqdm_notebook as tqdm
import glob
import numpy as np
import pandas as pd
import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score, KFold
from scipy.sparse import hstack
from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score
from datetime import datetime
import xgboost as xgb


LABEL_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train_original = pd.read_csv('../input/train.csv').fillna(' ')
test_original = pd.read_csv('../input/test.csv').fillna(' ')

oof_path = [
'fasttext_lstm1_cv_oof_0.988598_0.000409.csv',
'fasttext_lstm2_cv_oof_0.988600_0.000528.csv',
'fasttext_lstm3_cv_oof_0.983589_0.001340.csv',
'fasttext_lstm4_cudnn_cv_oof_0.987665_0.000706.csv',
'fasttext_lstm5_cudnn_cv_oof_0.987135_0.000816.csv',
'fasttext_lstm6_cudnn_cv_oof_0.987873_0.000603.csv',
'fasttext_lstm7_cudnn_cv_oof_0.988155_0.000630.csv',
'fasttext_lstm8_cudnn_cv_oof_0.983838_0.002173.csv',
'fasttext_lstm9_cudnn_cv_oof_0.983592_0.000722.csv',
'fasttext_lstm10_cudnn_cv_oof_0.989221_0.000313.csv',
'fasttext_lstm11_cudnn_cv_oof_0.989831_0.000165.csv',
'fasttext_lstm11_cudnn_prep_repeat_cv_oof_0.989970_0.000527.csv',
'fasttext_lstm12_cudnn_cv_oof_0.988968_0.000369.csv',
#'fasttext_lstm13_cudnn_cv_oof_0.978980_0.000754.csv',
#'fasttext_lstm13_cudnn_cv_oof_0.985434_0.000798.csv',
'fasttext_lstm16_cudnn_prep_repeat_cv_oof_0.989764_0.000310.csv',
'fasttext_lstm17_cudnn_prep_repeat_cv_oof_0.989660_0.000302.csv',
'fasttext_lstm18_cudnn_prep_repeat_cv_oof_0.986455_0.000197.csv',
'fasttext_lstm19_cudnn_prep_repeat_cv_oof_0.989378_0.000427.csv',
#'fasttext_conv1_cv_oof_0.974374_0.002278.csv',
'fasttext_correct_toxic_cv_oof_0.988551_0.000352.csv',
#'fasttext_features_cv_oof_0.984296_0.000462.csv',
'test_fasttext2_cv_oof_0.988522_0.000522.csv',
'tuned_LR1_cv_oof_0.985933_0.001310.csv',
'tuned_LR2_cv_oof_0.985977_0.001282.csv',
'tuned_LR3_cv_oof_0.986200_0.001215.csv',
'tuned_LR4_cv_oof_0.985634_0.001375.csv',
'tuned_LR5_cv_oof_0.985579_0.001347.csv',
'tuned_LR6_cv_oof_0.985522_0.001373.csv',
'tuned_LR7_cv_oof_0.986033_0.001198.csv',
'tuned_LR8_cv_oof_0.986323_0.001187.csv',
'tuned_LR9_cv_oof_0.986146_0.001234.csv',
'tuned_LR10_cv_oof_0.986133_0.001251.csv',
'tuned_LR11_cv_oof_0.984673_0.001157.csv',
'tuned_LR12_cv_oof_0.986285_0.001233.csv',
'fasttext_capsule1_cv_oof_0.975558_0.004659.csv',
'fasttext_capsule2_cv_oof_0.988845_0.000624.csv',
'fasttext_capsule3_cv_oof_0.989005_0.000365.csv',
'fasttext_capsule4_cv_oof_0.987758_0.001090.csv',
'fasttext_capsule5_cv_oof_0.988111_0.000818.csv',
'fasttext_capsule6_cv_oof_0.988748_0.000740.csv',
'fasttext_capsule7_cv_oof_0.988798_0.000352.csv',
'train_neptune.csv'
]

test_path = [
'fasttext_lstm1_cv_test.csv',
'fasttext_lstm2_cv_test.csv',
'fasttext_lstm3_cv_test.csv',
'fasttext_lstm4_cudnn_cv_test.csv',
'fasttext_lstm5_cudnn_cv_test.csv',
'fasttext_lstm6_cudnn_cv_test.csv',
'fasttext_lstm7_cudnn_cv_test.csv',
'fasttext_lstm8_cudnn_cv_test.csv',
'fasttext_lstm9_cudnn_cv_test.csv',
'fasttext_lstm10_cudnn_cv_test.csv',
'fasttext_lstm11_cudnn_cv_test.csv',
'fasttext_lstm11_cudnn_prep_repeat_cv_test.csv',
'fasttext_lstm12_cudnn_cv_test.csv',
#'fasttext_lstm13_cudnn_cv_test.csv',
'fasttext_lstm16_cudnn_prep_repeat_cv_test.csv',
'fasttext_lstm17_cudnn_prep_repeat_cv_test.csv',
'fasttext_lstm18_cudnn_prep_repeat_cv_test.csv',
'fasttext_lstm19_cudnn_prep_repeat_cv_test.csv',
#'fasttext_conv1_cv_test.csv',
'fasttext_correct_toxic_cv_test.csv',
#'fasttext_features_cv_test.csv',
'test_fasttext2_cv_test.csv',
'tuned_LR1_cv_test.csv',
'tuned_LR2_cv_test.csv',
'tuned_LR3_cv_test.csv',
'tuned_LR4_cv_test.csv',
'tuned_LR5_cv_test.csv',
'tuned_LR6_cv_test.csv',
'tuned_LR7_cv_test.csv',
'tuned_LR8_cv_test.csv',
'tuned_LR9_cv_test.csv',
'tuned_LR10_cv_test.csv',
'tuned_LR11_cv_test.csv',
'tuned_LR12_cv_test.csv',
'fasttext_capsule1_cv_test.csv',
'fasttext_capsule2_cv_test.csv',
'fasttext_capsule3_cv_test.csv',
'fasttext_capsule4_cv_test.csv',
'fasttext_capsule5_cv_test.csv',
'fasttext_capsule6_cv_test.csv',
'fasttext_capsule7_cv_test.csv',
'test_neptune.csv'
]


# Train
j = 0
for i in oof_path:
    tmp = pd.read_csv('../output/' + i)
    tmp = tmp.groupby('id').mean().reset_index()
    j_str = str(j)
    tmp.columns = ["id"] + (tmp.columns[1:]+'_'+j_str).tolist()
    if j == 0:
        train = tmp
    else:
        train = train.merge(tmp, how='left', on='id')
    j += 1

# Test
j = 0
for i in test_path:
    tmp = pd.read_csv('../output/' + i)
    tmp = tmp.groupby('id').mean().reset_index()
    j_str = str(j)
    tmp.columns = ["id"] + (tmp.columns[1:]+'_'+j_str).tolist()
    if j == 0:
        test = tmp
    else:
        test = test.merge(tmp, how='left', on='id')
    j += 1


train = train_original[['id']].merge(train, how='left', on='id')
test = test_original[['id']].merge(test, how='left', on='id')

train_features = train[train.columns.drop('id')]
test_features = test[test.columns.drop('id')]

target = train_original[LABEL_COLUMNS]

"""
Stacking
"""

folds = 5
scores = []
scores_classes = np.zeros((len(LABEL_COLUMNS), folds))

submission = pd.DataFrame.from_dict({'id': test['id']})
submission_oof = train_original[['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']]
#skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1001)
kf = KFold(n_splits=folds, random_state=4324455)


xgtest = xgb.DMatrix(test_features)

for j, (class_name) in enumerate(LABEL_COLUMNS):
    print(class_name)
    param = {}
    param['booster'] = 'gblinear'
    param['objective'] = 'rank:pairwise' #'rank:pairwise', 'binary:logistic'
    param['eta'] = 0.1
    param['max_depth'] = 10
    param['silent'] = 0
    param['eval_metric'] = 'auc'
    param['min_child_weight'] = 1
    param['gamma'] = 0.01
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8
    param['lambda'] = 0.001
    param['nthread'] = 6
    param['seed'] = 407 * (j+1)

    num_rounds = 500

    plst = list(param.items())

    avreal = target[class_name]
    lr_cv_sum = 0
    lr_pred = []
    lr_fpred = []
    lr_avpred = np.zeros(train.shape[0])

    for i, (train_index, val_index) in enumerate(kf.split(train_features)):
        X_train, X_val = train_features.iloc[train_index], train_features.iloc[val_index]
        y_train, y_val = target.loc[train_index], target.loc[val_index]

        xgtrain = xgb.DMatrix(X_train, label=y_train[class_name])
        xgval = xgb.DMatrix(X_val, label=y_val[class_name])
        watchlist = [ (xgtrain,'train'), (xgval, 'val') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist)

        scores_val = model.predict(xgval)
        lr_avpred[val_index] = scores_val
        lr_y_pred = model.predict(xgtest)
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

    submission[class_name] = lr_pred / folds
    submission_oof[class_name] = lr_avpred

auc_oof_mean = np.mean(np.mean(scores_classes, 1))
auc_oof_std = np.mean(np.std(scores_classes, 1))

saving_filename = 'stacking_cv'

print('\n Overall AUC: %.6f+/-%.6f' % (auc_oof_mean, auc_oof_std))
submission.to_csv('../output/{}_test.csv'.format(saving_filename), index=False)
submission_oof.to_csv('../output/{}_oof_{:.06f}_{:.06f}.csv'.format(saving_filename, auc_oof_mean, auc_oof_std), index=False)







from xgboost import XGBClassifier

estimator = XGBClassifier

"""
Toxic
"""
param_grid = dict(objective= ['rank:pairwise'],
                  eval_metric= ['auc'],
                  scale_pos_weight = [100],
                  n_estimators= [500],
                  learning_rate= [0.1],
                  max_depth= [2,3,4,5],
                  min_child_weight= [1,3,5,7],
                  gamma=[0.01,0.05,0.1,0.5],
                  subsample= [1.0,0.8,0.6],
                  colsample_bytree= [0.4,0.6,0.8,1.0], 
                  reg_lambda= [0.0,0.01,0.1,0.5,1.0], #1.0
                  reg_alpha= [0.0],
                  n_jobs=[12]
                 )

label_id = 0
nr_runs = 50
grid_sample = np.random.choice(make_grid(param_grid), nr_runs, replace=False)

grid_scores = []
for params in tqdm(grid_sample):
    scores, estimators, test_prediction = fit_cv(estimator, params, train, test, label_id, n_splits=10)  
#     plot_xgboost_learning_curve(estimators[-1])
#     print(params)
    print('mean {} std {}'.format(np.mean(scores), np.std(scores)))
    grid_scores.append((params, np.mean(scores)))
    
best_params = sorted(grid_scores, key= lambda x: x[1])[-1]
print(best_params)

from xgboost import XGBClassifier

estimator = XGBClassifier

label_id = 0

scores, estimators, test_prediction = fit_cv(estimator, best_params[0], train, test, label_id, n_splits=10)  
valid_scores['toxic'] = scores
predictions_test['toxic'] = test_prediction
plot_xgboost_learning_curve(estimators[-1])
print(params)
print('mean {} std {}'.format(np.mean(scores), np.std(scores)))


"""
Severe
"""
estimator = XGBClassifier
    
param_grid = dict(objective= ['rank:pairwise'],
                  eval_metric= ['auc'],
                  scale_pos_weight = [100],
                  n_estimators= [500],
                  learning_rate= [0.1],
                  max_depth= [2,3,4,5],
                  min_child_weight= [1,3,5,7],
                  gamma=[0.01,0.05,0.1,0.5],
                  subsample= [1.0,0.8,0.6],
                  colsample_bytree= [0.4,0.6,0.8,1.0], 
                  reg_lambda= [0.0,0.01,0.1,0.5,1.0], #1.0
                  reg_alpha= [0.0],
                  n_jobs=[12]
                 )

label_id = 1

nr_runs = 50
grid_sample = np.random.choice(make_grid(param_grid), nr_runs, replace=False)

grid_scores = []
for params in tqdm(grid_sample):
    scores, estimators, test_prediction = fit_cv(estimator, params, train, test, label_id, n_splits=10)  
#     plot_xgboost_learning_curve(estimators[-1])
#     print(params)
    print('mean {} std {}'.format(np.mean(scores), np.std(scores)))
    grid_scores.append((params, np.mean(scores)))
    
best_params = sorted(grid_scores, key= lambda x: x[1])[-1]
print(best_params)

label_id = 1

scores, estimators, test_prediction = fit_cv(estimator, best_params[0], train, test, label_id, n_splits=10)  
valid_scores['severe_toxic'] = scores
predictions_test['severe_toxic'] = test_prediction
plot_xgboost_learning_curve(estimators[-1])
print(params)
print('mean {} std {}'.format(np.mean(scores), np.std(scores)))

"""
obscene
"""
estimator = XGBClassifier
    
param_grid = dict(objective= ['rank:pairwise'],
                  eval_metric= ['auc'],
                  scale_pos_weight = [18],
                  n_estimators= [500],
                  learning_rate= [0.1],
                  max_depth= [2,3,4,5],
                  min_child_weight= [1,3,5,7],
                  gamma=[0.01,0.05,0.1,0.5],
                  subsample= [1.0,0.8,0.6],
                  colsample_bytree= [0.4,0.6,0.8,1.0], 
                  reg_lambda= [0.0,0.01,0.1,0.5,1.0], #1.0
                  reg_alpha= [0.0],
                  n_jobs=[12]
                 )

label_id = 2

nr_runs = 50
grid_sample = np.random.choice(make_grid(param_grid), nr_runs, replace=False)

grid_scores = []
for params in tqdm(grid_sample):
    scores, estimators, test_prediction = fit_cv(estimator, params, train, test, label_id, n_splits=10)  
#     plot_xgboost_learning_curve(estimators[-1])
#     print(params)
    print('mean {} std {}'.format(np.mean(scores), np.std(scores)))
    grid_scores.append((params, np.mean(scores)))
    
best_params = sorted(grid_scores, key= lambda x: x[1])[-1]
print(best_params)

label_id = 2

scores, estimators, test_prediction = fit_cv(estimator, best_params[0], train, test, label_id, n_splits=10)  
valid_scores['obscene'] = scores
predictions_test['obscene'] = test_prediction
plot_xgboost_learning_curve(estimators[-1])
print(params)
print('mean {} std {}'.format(np.mean(scores), np.std(scores)))

"""
threat
"""
estimator = XGBClassifier
    
param_grid = dict(objective= ['rank:pairwise'],
                  eval_metric= ['auc'],
                  scale_pos_weight = [331],
                  n_estimators= [500],
                  learning_rate= [0.1],
                  max_depth= [2,3,4,5],
                  min_child_weight= [1,3,5,7],
                  gamma=[0.01,0.05,0.1,0.5],
                  subsample= [1.0,0.8,0.6],
                  colsample_bytree= [0.4,0.6,0.8,1.0], 
                  reg_lambda= [0.0,0.01,0.1,0.5,1.0], #1.0
                  reg_alpha= [0.0],
                  n_jobs=[12]
                 )

label_id = 3

nr_runs = 50
grid_sample = np.random.choice(make_grid(param_grid), nr_runs, replace=False)

grid_scores = []
for params in tqdm(grid_sample):
    scores, estimators, test_prediction = fit_cv(estimator, params, train, test, label_id, n_splits=10)  
#     plot_xgboost_learning_curve(estimators[-1])
#     print(params)
    print('mean {} std {}'.format(np.mean(scores), np.std(scores)))
    grid_scores.append((params, np.mean(scores)))
    
best_params = sorted(grid_scores, key= lambda x: x[1])[-1]
print(best_params)

label_id = 3

scores, estimators, test_prediction = fit_cv(estimator, best_params[0], train, test, label_id, n_splits=10)  
valid_scores['threat'] = scores
predictions_test['threat'] = test_prediction
plot_xgboost_learning_curve(estimators[-1])
print(params)
print('mean {} std {}'.format(np.mean(scores), np.std(scores)))

"""
insult
"""
estimator = XGBClassifier
    
param_grid = dict(objective= ['rank:pairwise'],
                  eval_metric= ['auc'],
                  scale_pos_weight = [20],
                  n_estimators= [500],
                  learning_rate= [0.1],
                  max_depth= [2,3,4,5],
                  min_child_weight= [1,3,5,7],
                  gamma=[0.01,0.05,0.1,0.5],
                  subsample= [1.0,0.8,0.6],
                  colsample_bytree= [0.4,0.6,0.8,1.0], 
                  reg_lambda= [0.0,0.01,0.1,0.5,1.0], #1.0
                  reg_alpha= [0.0],
                  n_jobs=[12]
                 )

label_id = 4

nr_runs = 50
grid_sample = np.random.choice(make_grid(param_grid), nr_runs, replace=False)

grid_scores = []
for params in tqdm(grid_sample):
    scores, estimators, test_prediction = fit_cv(estimator, params, train, test, label_id, n_splits=10)  
#     plot_xgboost_learning_curve(estimators[-1])
    print(params)
    print('mean {} std {}'.format(np.mean(scores), np.std(scores)))
    grid_scores.append((params, np.mean(scores)))
    
best_params = sorted(grid_scores, key= lambda x: x[1])[-1]
print(best_params)

label_id = 4

scores, estimators, test_prediction = fit_cv(estimator, best_params[0], train, test, label_id, n_splits=10)  
valid_scores['insult'] = scores
predictions_test['insult'] = test_prediction
plot_xgboost_learning_curve(estimators[-1])
print(params)
print('mean {} std {}'.format(np.mean(scores), np.std(scores)))

"""
identity_hate
"""
estimator = XGBClassifier
    
param_grid = dict(objective= ['rank:pairwise'],
                  eval_metric= ['auc'],
                  scale_pos_weight = [112],
                  n_estimators= [500],
                  learning_rate= [0.1],
                  max_depth= [2,3,4,5],
                  min_child_weight= [1,3,5,7],
                  gamma=[0.01,0.05,0.1,0.5],
                  subsample= [1.0,0.8,0.6],
                  colsample_bytree= [0.4,0.6,0.8,1.0], 
                  reg_lambda= [0.0,0.01,0.1,0.5,1.0], #1.0
                  reg_alpha= [0.0],
                  n_jobs=[12]
                 )

label_id = 5

nr_runs = 50
grid_sample = np.random.choice(make_grid(param_grid), nr_runs, replace=False)

grid_scores = []
for params in tqdm(grid_sample):
    scores, estimators, test_prediction = fit_cv(estimator, params, train, test, label_id, n_splits=10)  
#     plot_xgboost_learning_curve(estimators[-1])
#     print(params)
    print('mean {} std {}'.format(np.mean(scores), np.std(scores)))
    grid_scores.append((params, np.mean(scores)))
    
best_params = sorted(grid_scores, key= lambda x: x[1])[-1]
print(best_params)

label_id = 5

scores, estimators, test_prediction = fit_cv(estimator, best_params[0], train, test, label_id, n_splits=10)  
valid_scores['identity_hate'] = scores
predictions_test['identity_hate'] = test_prediction
plot_xgboost_learning_curve(estimators[-1])
print(params)
print('mean {} std {}'.format(np.mean(scores), np.std(scores)))


"""
combine
"""
final_cv_score = []
for label, scores in valid_scores.items():
    final_cv_score.append(np.mean(scores))
    print(label, np.mean(scores),np.std(scores))
print(np.mean(final_cv_score))

"""
"""
def oof_scaling(predictions):
    predictions = pd.DataFrame(predictions)
    predictions = (1 + predictions.rank().values) / (predictions.shape[0] + 1)
    return predictions

for l, label in enumerate(LABEL_COLUMNS):
    predictions = predictions_test[label]
    for fold_predictions in predictions:
        fold_predictions = oof_scaling(fold_predictions)
#         fold_predictions_logit = np.log((fold_predictions + 1e-5) / (1 - fold_predictions + 1e-5))
#         sns.distplot(predictions_logit, hist=False)
        sns.distplot(fold_predictions, hist=False)
    plt.title(label)
    plt.show()

combined_predictions = {}
for label, predictions in predictions_test.items():
    oof_predictions = [oof_scaling(fold_predictions) for fold_predictions in predictions]
    oof_predictions_mean = np.mean(np.stack(oof_predictions, axis=-1),axis=-1).reshape(-1)
    combined_predictions[label] = oof_predictions_mean.tolist()
combined_predictions = pd.DataFrame(combined_predictions)

combined_predictions.head()

ENSEMBLE_SUBMISSION_PATH = '../output/xgboost_submission.csv'

submission = pd.read_csv('../input/sample_submission.csv')
submission[LABEL_COLUMNS] = combined_predictions[LABEL_COLUMNS].values 
submission.to_csv(ENSEMBLE_SUBMISSION_PATH, index=None)
submission.head()
