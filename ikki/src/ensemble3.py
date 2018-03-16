import pandas as pd

sample_submission = pd.read_csv('../input/sample_submission.csv')
test_fasttext1 = pd.read_csv('../output/9868model_blend_it_all_5_5.csv')
#test_fasttext2 = pd.read_csv('/Users/ikki.tanaka/Downloads/submission.csv')
test_fasttext3 = pd.read_csv('../output/test.csv')


labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
for label in labels:
    print(label)
    print(np.corrcoef([test_fasttext1[label].rank(pct=True), test_fasttext3[label].rank(pct=True)]))

for label in labels:
    sample_submission[label] = test_fasttext1[label].rank(pct=True) * 0.5 + test_fasttext3[label].rank(pct=True) * 0.5

sample_submission.to_csv('../output/9868model_blend_it_all_5_5.csv', index=False)

