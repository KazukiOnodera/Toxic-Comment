"""
連続する文 or 単語をユニークにする
---
I FUCK YOU I FUCK YOU I FUCK YOU => I FUCK YOU
I FUCK FUCK YOU => I FUCK YOU
---
2文字連続する文字は変換しない
correct => correct
corrrect => corect
"""
import re
import pandas as pd
import tqdm

print("Loading train/test data")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train_feat = pd.read_pickle('../data/102_train.p')
test_feat = pd.read_pickle('../data/102_test.p')

train = train.merge(train_feat, how='left', on='id')
test = test.merge(test_feat, how='left', on='id')

for i in tqdm(range(3, 300)):
    pattern = r'([\w\W]{' + f'{i}' + r'})\1{1,}'
    p = re.compile(pattern)
    rm_repeat_func = lambda x: p.sub(r'\1', x) if len(x) < i else x
    train['comment_text'] = train.comment_text.apply(rm_repeat_func)
    test['comment_text'] = test.comment_text.apply(rm_repeat_func)

train['comment_len_rm_repeat'] = train['comment_text'].apply(len)
test['comment_len_rm_repeat'] = test['comment_text'].apply(len)

train['diff_comment_len_repeat'] = train.comment_len - train.comment_len_rm_repeat
test['diff_comment_len_repeat'] = test.comment_len - test.comment_len_rm_repeat


train = train[['id', 'comment_text']]
test = test[['id', 'comment_text']]

train.to_pickle('../data/train_rm_repeat.p')
test.to_pickle('../data/test_rm_repeat.p')

