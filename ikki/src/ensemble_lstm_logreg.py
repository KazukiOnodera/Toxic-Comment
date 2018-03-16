import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

ensemble_name = 'rank_ensemble1'

def rank_avg(preds_path_list, save_name, weights=None, pred_type='train'):
    sample_submission = pd.read_csv('../input/sample_submission.csv')
    preds_list = [pd.read_csv('../output/' + pred) for pred in preds_path_list]
    labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    
    if weights is not None:
        assert len(weights) == len(preds_path_list)

    if weights is None:
        weights = np.ones(len(preds_path_list))

    print('correlation')
    for label in labels:
        print(label)
        print(np.corrcoef([pred_df[label].rank(pct=True) for pred_df in preds_list]))

    if pred_type == 'train':
        print('AUC of rank averaging in train')
        train = pd.read_csv('../input/train.csv')
        oof_preds = train[['id'] + labels].copy()
        for label in labels:
            oof_preds[label] = np.sum([pred_df[label].rank(pct=True) * weight for pred_df, weight in zip(preds_list, weights)], axis=0)
            oof_preds[label] /= np.sum(weights)
        auc_val = roc_auc_score(train[labels].values, oof_preds[labels].values, average=None, sample_weight=None)
        print('Averaged Each AUC: {}'.format(auc_val))
        print('Averaged AUC: {}+/-{}'.format(np.mean(auc_val), np.std(auc_val)))

        print('saving at {}'.format('../output/{}_{}_oof_{:.06f}_{:.06f}.csv'.format(ensemble_name, save_name, np.mean(auc_val), np.std(auc_val))))
        oof_preds.to_csv('../output/{}_{}_oof_{:.06f}_{:.06f}.csv'.format(ensemble_name, save_name, np.mean(auc_val), np.std(auc_val)), index=False)

    elif pred_type == 'test':
        print('rank averaging')
        for label in labels:
            sample_submission[label] = np.sum([pred_df[label].rank(pct=True) * weight for pred_df, weight in zip(preds_list, weights)], axis=0)
            sample_submission[label] /= np.sum(weights)

        print('saving at {}'.format('../output/{}_{}_test.csv'.format(ensemble_name, save_name)))
        sample_submission.to_csv('../output/{}_{}_test.csv'.format(ensemble_name, save_name), index=False)


"""
LSTMのensemble
"""
# val
val_pred_list = [
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
'fasttext_lstm13_cudnn_cv_oof_0.978980_0.000754.csv',
'fasttext_lstm13_cudnn_cv_oof_0.985434_0.000798.csv',
'fasttext_lstm16_cudnn_prep_repeat_cv_oof_0.989764_0.000310.csv',
'fasttext_lstm17_cudnn_prep_repeat_cv_oof_0.989660_0.000302.csv',
'fasttext_lstm18_cudnn_prep_repeat_cv_oof_0.986455_0.000197.csv',
'fasttext_lstm19_cudnn_prep_repeat_cv_oof_0.989378_0.000427.csv',
'fasttext_conv1_cv_oof_0.974374_0.002278.csv',
'fasttext_correct_toxic_cv_oof_0.988551_0.000352.csv',
#'fasttext_features_cv_oof_0.984296_0.000462.csv',
'test_fasttext2_cv_oof_0.988522_0.000522.csv',
]

rank_avg(val_pred_list, save_name='lstm')

# test
test_pred_list = [
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
'fasttext_lstm13_cudnn_cv_test.csv',
'fasttext_lstm16_cudnn_prep_repeat_cv_test.csv',
'fasttext_lstm17_cudnn_prep_repeat_cv_test.csv',
'fasttext_lstm18_cudnn_prep_repeat_cv_test.csv',
'fasttext_lstm19_cudnn_prep_repeat_cv_test.csv',
'fasttext_conv1_cv_test.csv',
'fasttext_correct_toxic_cv_test.csv',
#'fasttext_features_cv_test.csv',
'test_fasttext2_cv_test.csv',
]

rank_avg(test_pred_list, save_name='lstm', pred_type='test')



"""
LogReg
"""
# val
val_pred_list = [
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
]

rank_avg(val_pred_list, save_name='logreg')

# test
test_pred_list = [
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
]

rank_avg(test_pred_list, save_name='logreg', pred_type='test')

"""
LSTMとLogReg
"""
# val
val_pred_list = [
'rank_ensemble1_lstm_oof_0.990623_0.002306.csv',
'rank_ensemble1_logreg_oof_0.986189_0.004719.csv'
]

rank_avg(val_pred_list, save_name='lstm_logreg', weights=[8.5, 1.5])

# test
test_pred_list = [
'rank_ensemble1_lstm_test.csv',
'rank_ensemble1_logreg_test.csv'
]

rank_avg(test_pred_list, save_name='lstm_logreg', weights=[8.5, 1.5], pred_type='test')

"""
hoxosh
"""
# test
test_pred_list = [
'rank_ensemble1_lstm_logreg_test.csv',
'ikki_20_ensemble_0313.csv'
]

rank_avg(test_pred_list, save_name='lstm_logreg_hoxosh', weights=[1, 1], pred_type='test')
