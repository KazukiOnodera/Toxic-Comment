
import pandas as pd

train = pd.read_pickle('../data/train2.p')
test = pd.read_pickle('../data/test2.p')


# length features
def main(df):
    
    df['comment_len'] = df.comment_text.map(len)
    comment_split = df.comment_text.map(lambda x: x.split())
    df['word_cnt'] = comment_split.map(lambda x: len(x))
    df['word_cnt_unq'] = comment_split.map(lambda x: len(set(x)) )
    df['word_unq_raito'] = df['word_cnt_unq'] / df['word_cnt']
    df['word_max_len'] = comment_split.map(lambda x: max(map(len, set(x))) )
    
    col = ['id', 'comment_len', 'word_cnt', 'word_cnt_unq', 'word_max_len']
    return df[col]

# ==============================================================
# main
# ==============================================================

main(train).to_pickle('../data/102_train.p')
main(test).to_pickle('../data/102_test.p')

