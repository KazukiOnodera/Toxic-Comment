
import pandas as pd
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))
from nltk import stem
pt = stem.PorterStemmer()
import re
from collections import Counter
from tqdm import tqdm


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


def tokenizer(s):
    s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
    s = re.sub(r"\s([a-z]+)([A-Z])", r" \1 \2", s)
    s = s.replace("$"," ")
    s = s.replace("?"," ")
    s = s.replace("!"," ")
    s = s.replace("-"," ")
    s = s.replace("//","/")
    s = s.replace("..",".")
    s = s.replace(" / "," ")
    s = s.replace(" \\ "," ")
    s = s.replace("("," ")
    s = s.replace(")"," ")
    s = s.replace("{"," ")
    s = s.replace("}"," ")
    s = s.replace(":"," ")
    s = s.replace(".","  ")
    s = s.replace("…","  ")
    s = s.replace(",","  ")
    s = s.replace("''", "  ")
    s = s.replace('==', ' ')
    
    s = s.replace('"',"")
    s = s.replace("'","")
    s = s.lower()
    return s

def remove_stop(txt):
    return ' '.join([w for w in txt.split() if w not in stopWords])

def get_stem(txt):
    try:
        return ' '.join(list(map(pt.stem, txt.split()))).strip()
    except:
        print(txt)
        return txt

train_ = train.copy()
train_.comment_text = train_.comment_text.map(tokenizer).map(remove_stop).map(get_stem)

test_ = test.copy()
test_.comment_text = test_.comment_text.map(tokenizer).map(remove_stop).map(get_stem)

train_.to_pickle('../data/train1.p')
test_.to_pickle('../data/test1.p')

counter = Counter()
cnt = 0; pool = []
for s in tqdm(train_.comment_text):
    cnt +=1
    counter += Counter(s.split())
    if cnt % 5000 == 0 and cnt>0:
        pool.append(counter)
        counter = Counter()
pool.append(counter)

counter = Counter()
for p in pool:
    counter += p

df = pd.DataFrame(counter.most_common(), columns=['word', 'freq'])
df.to_pickle('../data/train1_freq.p')


counter = Counter()
cnt = 0; pool = []
for s in tqdm(test_.comment_text):
    cnt +=1
    counter += Counter(s.split())
    if cnt % 5000 == 0 and cnt>0:
        pool.append(counter)
        counter = Counter()
pool.append(counter)

counter = Counter()
for p in pool:
    counter += p

df = pd.DataFrame(counter.most_common(), columns=['word', 'freq'])
df.to_pickle('../data/test1_freq.p')




# only tokenizer

train_ = train.copy()
train_.comment_text = train_.comment_text.map(tokenizer)

test_ = test.copy()
test_.comment_text = test_.comment_text.map(tokenizer)


train_.to_pickle('../data/train2.p')
test_.to_pickle('../data/test2.p')


counter = Counter()
cnt = 0; pool = []
for s in tqdm(train_.comment_text):
    cnt +=1
    counter += Counter(s.split())
    if cnt % 5000 == 0 and cnt>0:
        pool.append(counter)
        counter = Counter()
pool.append(counter)

counter = Counter()
for p in pool:
    counter += p

df = pd.DataFrame(counter.most_common(), columns=['word', 'freq'])
df.to_pickle('../data/train2_freq.p')


counter = Counter()
cnt = 0; pool = []
for s in tqdm(test_.comment_text):
    cnt +=1
    counter += Counter(s.split())
    if cnt % 5000 == 0 and cnt>0:
        pool.append(counter)
        counter = Counter()
pool.append(counter)

counter = Counter()
for p in pool:
    counter += p

df = pd.DataFrame(counter.most_common(), columns=['word', 'freq'])
df.to_pickle('../data/test2_freq.p')


