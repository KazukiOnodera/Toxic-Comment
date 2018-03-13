"""
0.9853モデル
"""
'fuck off thank you!'
array([[9.9188083e-01, 1.5183511e-01, 9.9372840e-01, 2.9854421e-04,
        2.9169053e-01, 1.3605916e-03]], dtype=float32)

'f**k off thank you!'
array([[9.0031958e-01, 5.4308526e-02, 9.6262074e-01, 3.0485471e-04,
        6.0572438e-02, 5.2989781e-04]], dtype=float32)

'**** off thank you!'
array([[0.2728691 , 0.00723547, 0.21662225, 0.00031662, 0.01985802,
        0.00055534]], dtype=float32)

'**** *** thank you!'
array([[1.0872832e-01, 2.1058016e-03, 5.0414674e-02, 7.0800590e-05,
        8.3896676e-03, 4.1742399e-04]], dtype=float32)

model = keras.load(...)
model.predict([text_to_vector('ikki tanaka'.split())])

"""
correlation
"""
from sklearn.metrics import roc_auc_score
import numpy as np
import pandas as pd

train = pd.read_csv('../input/train.csv')

""" CV """
lstm6_cu = pd.read_csv('../output/fasttext_lstm6_cudnn_cv_oof_0.987873_0.000603.csv') # PL score 0.9829
lstm5_cu = pd.read_csv('../output/fasttext_lstm5_cudnn_cv_oof_0.987135_0.000816.csv') # PL score 0.9829
lstm4_cu = pd.read_csv('../output/fasttext_lstm4_cudnn_cv_oof_0.987665_0.000706.csv') # PL score 0.9829
lstm1 = pd.read_csv('../output/fasttext_lstm1_cv_oof_0.988598_0.000409.csv') # PL score 0.9829

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for label in labels:
    print(label)
    print(np.corrcoef([lstm1[label].rank(pct=True), lstm4_cu[label].rank(pct=True), lstm5_cu[label].rank(pct=True), lstm6_cu[label].rank(pct=True)]))

submission = pd.DataFrame()
submission['id'] = lstm1['id']
for label in labels:
    submission[label] = lstm1[label].rank(pct=True) * 0.5 + lstm4_cu[label].rank(pct=True) * 0.2 + lstm5_cu[label].rank(pct=True) * 0.1 + lstm6_cu[label].rank(pct=True) * 0.2

auc_val = roc_auc_score(train[labels], submission[labels], average=None, sample_weight=None)
auc_val_mean = np.mean(auc_val)
auc_val_std = np.std(auc_val)
print('Averaged AUC of validation: {}+{}'.format(auc_val_mean, auc_val_std))

""" Test """
lstm6_cu = pd.read_csv('../output/fasttext_lstm6_cudnn_cv_test.csv') # PL score 0.9829
lstm5_cu = pd.read_csv('../output/fasttext_lstm5_cudnn_cv_test.csv') # PL score 0.9829
lstm4_cu = pd.read_csv('../output/fasttext_lstm4_cudnn_cv_test.csv') # PL score 0.9829
lstm1 = pd.read_csv('../output/fasttext_lstm1_cv_test.csv') # PL score 0.9829
hight_of_blend_v2 = pd.read_csv('../output/hight_of_blend_v2.csv')
corr_blend = pd.read_csv('../output/corr_blend.csv')
fasttext_correct = pd.read_csv('../output/fasttext_correct_toxic_cv_test.csv')
test_fasttext2 = pd.read_csv('../output/test_fasttext2_cv_test.csv')

# The value of an ensemble is (a) the individual scores of the models and
# (b) their correlation with one another. We want multiple individually high
# scoring models that all have low correlations. Based on this analysis, it
# looks like these kernels have relatively low correlations and will blend to a
# much higher score.
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for label in labels:
    print(label)
    print(np.corrcoef([lstm1[label].rank(pct=True),
                       lstm4_cu[label].rank(pct=True),
                       lstm5_cu[label].rank(pct=True),
                       lstm6_cu[label].rank(pct=True),
                       hight_of_blend_v2[label].rank(pct=True),
                       corr_blend[label].rank(pct=True),
                       fasttext_correct[label].rank(pct=True),
                       test_fasttext2[label].rank(pct=True)]))

submission = pd.DataFrame()
submission['id'] = lstm1['id']
for label in labels:
    submission[label] = lstm1[label].rank(pct=True) * 0.10 + \
                        lstm4_cu[label].rank(pct=True) * 0.04 + \
                        lstm5_cu[label].rank(pct=True) * 0.02 + \
                        lstm6_cu[label].rank(pct=True) * 0.04 + \
                        fasttext_correct[label].rank(pct=True) * 0.10 + \
                        hight_of_blend_v2[label].rank(pct=True) * 0.3 + \
                        corr_blend[label].rank(pct=True) * 0.3 + \
                        test_fasttext2[label].rank(pct=True) * 0.10

submission.to_csv('../output/ensemble_test3.csv', index=False)
