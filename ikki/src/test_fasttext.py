"""
https://www.kaggle.com/mschumacher/using-fasttext-models-for-robust-embeddings/notebook
"""
import os
import re
import math
import numpy as np
import pandas as pd
from fastText import load_model
import unidecode
import logging

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, GRU, Embedding, Dropout, Activation, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D, TimeDistributed
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers

from sklearn.metrics import roc_auc_score


window_length = 200 # The amount of words we look at per example. Experiment with this.

"""
Remove accent marks
"""
def remove_accent_before_tokens(sentences):
    res = unidecode.unidecode(sentences)
    return(res)

"""
Expanding contraction 
"""
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
    # Expand contractions
    s = expand_contractions(s, CONTRACTION_MAP)
    # Remove accent marks
    s = remove_accent_before_tokens(s)
    # Replace ips
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    # Isolate punctuation
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Remove new lines
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')
    return s

print('\nLoading data')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train['comment_text'] = train['comment_text'].fillna('_empty_')
test['comment_text'] = test['comment_text'].fillna('_empty_')


# Loading FT model
classes = [
    'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'
]

print('\nLoading FT model')
ft_model = load_model('../external_data/pretrained/fasttext/wiki/wiki.en.bin')
n_features = ft_model.get_dimension()

def text_to_vector(text):
    """
    Given a string, normalizes it, then splits it into words and finally converts
    it to a sequence of word vectors.
    ---
    window_lengthよりも大きい文章のときスタート位置をランダムにする機能追加
    """
    text = normalize(text)
    words = text.split()
    #TODO: remove stop words(https://www.kaggle.com/saxinou/nlp-01-preprocessing-data)
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

# Creating Training/Testing
# Split the dataset:
split_index = round(len(train) * 0.9)
shuffled_train = train.sample(frac=1, random_state=0)
df_train = shuffled_train.iloc[:split_index]
df_val = shuffled_train.iloc[split_index:]

# Convert validation set to fixed array
x_val = df_to_data(df_val)
y_val = df_val[classes].values

x_test = df_to_data(test)


def data_generator(df, batch_size):
    """
    Given a raw dataframe, generates infinite batches of FastText vectors.
    """
    batch_i = 0 # Counter inside the current batch vector
    batch_x = None # The current batch's x data
    batch_y = None # The current batch's y data
    
    while True: # Loop forever
        df = df.sample(frac=1) # Shuffle df each epoch
        
        for i, row in df.iterrows():
            comment = row['comment_text']
            
            if batch_x is None:
                batch_x = np.zeros((batch_size, window_length, n_features), dtype='float32')
                batch_y = np.zeros((batch_size, len(classes)), dtype='float32')
                
            batch_x[batch_i] = text_to_vector(comment)
            batch_y[batch_i] = row[classes].values
            batch_i += 1

            if batch_i == batch_size:
                # Ready to yield the batch
                yield batch_x, batch_y
                batch_x = None
                batch_y = None
                batch_i = 0



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

def build_model(logdir='.'):
    # Bidirectional-LSTM
    if logdir is not None and os.path.exists(logdir):
        tb_cb = TensorBoard(log_dir='.', histogram_freq=0, write_graph=True)

    inp = Input(shape=(window_length, 300))
    x = Bidirectional(LSTM(50, return_sequences=True, dropout=0.1, recurrent_dropout=0.1))(inp)
    x = GlobalMaxPool1D()(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        #self.input_spec = [InputSpec(ndim=3)]
        super(AttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = self.init((input_shape[-1],1))
        #self.W = self.init((input_shape[-1],))
        #self.input_spec = [InputSpec(shape=input_shape)]
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.dot(x, self.W))

        ai = K.exp(eij)
        weights = ai/K.expand_dims(K.sum(ai, axis=1), 1)
        #K.sum(ai, axis=1).dimshuffle(0,'x')

        weighted_input = x*K.expand_dims(weights, 2)
        #weights.dimshuffle(0,1,'x')
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        print(input_shape)
        return (input_shape[0], input_shape[-1])

def build_attention_model(logdir='attention'):
    # Bidirectional-LSTM
    if logdir is not None:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        tb_cb = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True)

    inp = Input(shape=(window_length, 300))
    l_lstm = Bidirectional(GRU(50, return_sequences=True))(inp)
    l_dense = TimeDistributed(Dense(50))(l_lstm)
    #l_att = AttLayer()(l_dense)
    sentEncoder = Model(inputs=inp, outputs=l_dense)

    #review_input = Input(shape=(window_length, 300), dtype='float32')
    #review_encoder = TimeDistributed(sentEncoder)(l_att)
    l_lstm_sent = Bidirectional(GRU(50, return_sequences=True))(l_dense)
    l_dense_sent = TimeDistributed(Dense(50))(l_lstm_sent)
    #l_att_sent = AttLayer()(l_dense_sent)
    x = GlobalMaxPool1D()(l_dense_sent)
    x = Dense(128, activation="elu")(x)
    x = Dropout(0.5)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3, amsgrad=True), metrics=['accuracy'])

    return model


# Build model
model = build_attention_model()
#model = build_model()

batch_size = 256
training_steps_per_epoch = math.ceil(len(df_train) / batch_size)
training_generator = data_generator(df_train, batch_size)

ival = IntervalEvaluation(validation_data=(x_val, y_val), interval=1)

# Ready to start training:
callback_history = model.fit_generator(
    training_generator,
    steps_per_epoch=training_steps_per_epoch,
    epochs=5,
    validation_data=(x_val, y_val),
    #use_multiprocessing=True,
    callbacks=[ival]
)

x_val_pred = model.predict(x_val, batch_size=1024, verbose=1)
auc_val = roc_auc_score(y_val, x_val_pred, average=None, sample_weight=None)
print('Averaged AUC of validation: {}+{}'.format(np.mean(auc_val), np.std(auc_val)))

y_test = model.predict(x_test, batch_size=1024, verbose=1)
sample_submission = pd.read_csv('../input/sample_submission.csv')
sample_submission[classes] = y_test

sample_submission.to_csv('../output/test_fasttext_val0.9893_0.0037.csv', index=False)
