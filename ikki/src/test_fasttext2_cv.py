"""
参考
https://www.kaggle.com/mschumacher/using-fasttext-models-for-robust-embeddings/notebook

TODO: module化する
1. raw data -> preprocessing data
2. preprocessing data -> cross-validation -> prediction of out-of-fold and testset
3. stacking...
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
from keras.layers import Bidirectional, GlobalMaxPool1D, TimeDistributed, Permute, Reshape, Lambda, RepeatVector, Multiply, Concatenate

from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.optimizers import SGD, RMSprop, Adam
from keras.callbacks import Callback, EarlyStopping, TensorBoard, ModelCheckpoint
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


window_length = 250 # The amount of words we look at per example. Experiment with this.

"""
Preprocessing functions
"""
# Remove accent marks
def remove_accent_before_tokens(sentences):
    res = unidecode.unidecode(sentences)
    return(res)

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
    # Special preprocessing
    s = s.replace('ı', 'i')
    # Expand contractions
    s = expand_contractions(s, CONTRACTION_MAP)
    # Remove accent marks
    #s = remove_accent_before_tokens(s)
    # Replace ips
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    # Isolate punctuation
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s).replace('\1', '')
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

"""
Util functions for keras
"""
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

"""
Function to generate training data
"""
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

def data_generator_for_test(df, batch_size):
    """
    Given a raw dataframe, generates infinite batches of FastText vectors.
    """
    df_length = len(df)
    batch_i = 0 # Counter inside the current batch vector
    batch_x = None # The current batch's x data
    batch_y = None # The current batch's y data

    for i, row in df.iterrows():
        comment = row['comment_text']

        if batch_x is None:
            batch_x = np.zeros((batch_size, window_length, n_features), dtype='float32')
            batch_y = np.zeros((batch_size, len(classes)), dtype='float32')

        batch_x[batch_i] = text_to_vector(comment)
        batch_y[batch_i] = row[classes].values
        batch_i += 1

        if batch_i == batch_size or i + 1 == df_length:
            # Ready to yield the batch
            yield batch_x, batch_y
            batch_x = None
            batch_y = None
            batch_i = 0

"""
Data loading
"""
print("Loading train/test data")
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# target classes
classes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

"""
Preprocessings
"""
# Replace nan value
#train['comment_text'] = train['comment_text'].fillna('_empty_')
#test['comment_text'] = test['comment_text'].fillna('_empty_')

# Replace miscellaneous characters
train['comment_text'] = train['comment_text'].str.replace('ı', 'i')
test['comment_text'] = test['comment_text'].str.replace('ı', 'i')


# Normalize comment_text (IMPLEMENTED IN GENERATOR)
#train['comment_text'] = train['comment_text'].apply(normalize)
#test['comment_text'] = test['comment_text'].apply(normalize)

# Split comment_text (IMPLEMENTED IN GENERATOR)
#train['comment_text'] = train['comment_text'].str.split()
#test['comment_text'] = test['comment_text'].str.split()

"""
Loading FT model
"""
print('Loading FT model')
ft_model = load_model('../external_data/pretrained/fasttext/wiki/wiki.en.bin')
# Embedding dimension
n_features = ft_model.get_dimension()


"""
Define models
"""
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

def attention_3d_block(inputs, name, single_attention_vector=False):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, window_length))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(window_length, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction'+name)(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec'+name)(a)
    output_attention_mul = Multiply(name='mul'+name)([inputs, a_probs])
    return output_attention_mul

def build_attention_model():
    # Bidirectional-GRU with Attention
    inp = Input(shape=(window_length, 300))

    # Attention before LSTM
    #attention_mul1 = attention_3d_block(inp, name='inp')

    l_lstm1 = Bidirectional(GRU(100, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(inp)

    # Attention after LSTM
    attention_mul1 = attention_3d_block(l_lstm1, name='l_lstm1')

    #l_lstm2 = Bidirectional(GRU(50, return_sequences=True, dropout=0.15, recurrent_dropout=0.15))(attention_mul1)
    #attention_mul2 = attention_3d_block(l_lstm2, name='l_lstm2')

    # Flatten attention vectors
    attention_mul1 = Flatten()(attention_mul1)

    inp_feat = Input(shape=(n_engineered_features,))
    concat_feat = Concatenate()([attention_mul1, inp_feat])
    x = Dropout(0.1)(concat_feat)
    #x = Dense(216, activation="elu")(x)
    #x = Dropout(0.4)(x)
    x = Dense(512, activation="elu")(x)
    x = Dropout(0.3)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=[inp, inp_feat], outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3, amsgrad=True), metrics=['accuracy'])

    return model

def build_lstm_stack_model(logdir='attention'):
    # Bidirectional-LSTM
    inp = Input(shape=(window_length, 300))
    l_lstm = Bidirectional(GRU(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(inp)
    l_dense = TimeDistributed(Dense(50))(l_lstm)
    sentEncoder = Model(inputs=inp, outputs=l_dense)

    l_lstm_sent = Bidirectional(GRU(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(l_dense)
    l_dense_sent = TimeDistributed(Dense(50))(l_lstm_sent)

    x = GlobalMaxPool1D()(l_dense_sent)
    x = Dense(256, activation="elu")(x)
    x = Dropout(0.5)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3, amsgrad=True), metrics=['accuracy'])

    return model


"""
Training and evaluating with cross-validation
"""
# Filename to save
saving_filename = 'test_fasttext6_cv'

# Define KFold and random state
random_state = 407
n_splits = 5
kf = KFold(n_splits=n_splits, random_state=random_state)

# Make test dataset to fixed array
#print('Converting test dataframe to array')
#x_test = df_to_data(test)

# Cross validation
# Prediction of out-of-fold data
pred_oof = pd.concat([train['id'], pd.DataFrame(np.zeros((train.shape[0], train.shape[1]-2)), columns=classes)], axis=1)
auc_pred_oof = []
# Prediction of test dataset
pred_test = pd.read_csv('../input/sample_submission.csv')
pred_test[classes] = 0.0  #> initialize


for fold_idx, (train_index, val_index) in enumerate(kf.split(train)):
    print('\nFold: {}'.format(fold_idx))
    # Train/validation dataset
    x_train, x_val = train.iloc[train_index], train.iloc[val_index]
    y_train, y_val = train[classes].iloc[train_index], train[classes].iloc[val_index]

    # Convert validation set to fixed array
    print('Converting validation dataframe to array')
    x_val = df_to_data(x_val)
    #validation_generator = data_generator_for_test(x_val, 1024)
    y_val = y_val.values

    # Build model
    print('Building model')
    model = build_lstm_stack_model()

    # Parameters
    batch_size = 256
    training_steps_per_epoch = math.ceil(len(x_train) / batch_size)
    training_generator = data_generator(x_train, batch_size)

    # Callbacks
    ival = IntervalEvaluation(validation_data=(x_val, y_val), interval=1)

    # Training
    print('Training model')
    callback_history = model.fit_generator(
                            training_generator,
                            steps_per_epoch=training_steps_per_epoch,
                            epochs=10,
                            validation_data=(x_val, y_val),
                            callbacks=[ival]
                            )

    # Predict at validation dataset
    y_val_pred = model.predict(x_val, batch_size=1024, verbose=1)
    # Asign results to dataframe
    pred_oof.loc[val_index, classes] = y_val_pred
    # Evaluate validation results
    auc_val = roc_auc_score(y_val, y_val_pred, average=None, sample_weight=None)
    auc_val_mean = np.mean(auc_val)
    auc_val_std = np.std(auc_val)
    auc_pred_oof.append([auc_val_mean, auc_val_std])
    print('Averaged AUC of validation: {}+{}'.format(auc_val_mean, auc_val_std))

    # Predict test dataset several times at random
    testing_generator = data_generator_for_test(test, 1024)
    pred_test_num = 10
    y_test = np.array([])
    for i in range(pred_test_num):
        if i == 0:
            y_test_pred = model.predict_generator(testing_generator)
            y_test = np.expand_dims(y_test_pred, 2)
        else:
            y_test_pred = model.predict_generator(testing_generator)
            y_test_pred = np.expand_dims(y_test_pred, 2)
            y_test = np.concatenate([y_test, y_test_pred], axis=2)
    y_test = y_test.max(2)

    # Asign results to dataframe (divide followed by sum)
    pred_test[classes] += y_test / n_splits

    # Save model every fold
    from keras.models import load_model
    model.save('../model/model_{}_{}.h5'.format(saving_filename, fold_idx))  # creates a HDF5 file 'my_model.h5'

# AUC of cross-validation
auc_pred_oof_fold = [auc_mean for auc_mean, auc_std in auc_pred_oof]
auc_pred_oof_mean = np.mean(auc_pred_oof_mean)
auc_pred_oof_std = np.std(auc_pred_oof_mean)

# Save result of cross-validation
pred_oof.to_csv('../output/{}_oof_{:.06f}_{:.06f}.csv'.format(saving_filename, auc_pred_oof_mean, auc_pred_oof_std), index=False)
pred_test.to_csv('../output/{}_test.csv'.format(saving_filename), index=False)

"""
sample_submission = pd.read_csv('../input/sample_submission.csv')
test_fasttext1 = pd.read_csv('../output/test_fasttext_val0.9893_0.0037.csv')
test_fasttext2 = pd.read_csv('../output/test_fasttext_val0.990713_0.003469.csv')
sample_submission[classes] = (test_fasttext1[classes] + test_fasttext2[classes]) / 2
sample_submission.to_csv('../output/avg_test_fasttext1_test_fasttext2.csv', index=False)
# fasttext1: 0.9831
# fasttext2: 0.9841
# fasttext1+2: 0.9847
"""
