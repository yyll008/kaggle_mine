import time
start_time = time.time()
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import sys, os, re, csv, codecs, numpy as np, pandas as pd
from keras.preprocessing import text, sequence
np.random.seed(32)
os.environ["OMP_NUM_THREADS"] = "4"
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam, RMSprop, Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import GRU, BatchNormalization, Conv1D, MaxPooling1D
import pickle
from importlib import reload
import logging
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from keras.initializers import he_uniform
from keras.layers import PReLU
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))

from project_utils import toxic_utils

pd.options.display.float_format = '{:,.8f}'.format
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows', 1000)

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch+1, score))

# train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
# test = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv")
# embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"

train_x = pd.read_csv("../input/cleaned-toxic-comments/train_preprocessed.csv").fillna(" ")
test_x = pd.read_csv("../input/cleaned-toxic-comments/test_preprocessed.csv").fillna(" ")
embedding_path = "../input/glove840b300dtxt/glove.840B.300d.txt"

embed_size = 300
max_features = 165000#100000
max_len = 250#150

train_x['comment_text'].fillna(' ')
test_x['comment_text'].fillna(' ')
y = train_x[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
train_x = train_x['comment_text'].str.lower()
test_x = test_x['comment_text'].str.lower()


# Vectorize text + Prepare GloVe Embedding
tokenizer = text.Tokenizer(num_words=max_features, lower=True)
tokenizer.fit_on_texts(list(train_x))

train_x = tokenizer.texts_to_sequences(train_x)
test_x = tokenizer.texts_to_sequences(test_x)

train_x = sequence.pad_sequences(train_x, maxlen=max_len)
test_x = sequence.pad_sequences(test_x, maxlen=max_len)

gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.25
rate_drop_dense = 0.28

def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


def get_coefs(row):
    row = row.strip().split()
    word, arr = " ".join(row[:-embed_size]), row[-embed_size:]
    return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(row) for row in open(embedding_path))


all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

from keras.layers import Dense, Embedding, Input
from keras.layers import Bidirectional, Dropout, CuDNNGRU, SpatialDropout1D
from keras.models import Model
from keras.optimizers import RMSprop


def build_model(lr = 0.0, lr_d = 0.0, units = 0, dr = 0.0):
    inp = Input(shape = (max_len,))
    x = Embedding(max_features, embed_size, weights = [embedding_matrix], trainable = False)(inp)
    
    x = SpatialDropout1D(dr)(x)
    x11 = Bidirectional(GRU(60, return_sequences=True))(x)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    
    x1 = Conv1D(60, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x11)
    avg_pool11 = GlobalAveragePooling1D()(x1)
    max_pool11 = GlobalMaxPooling1D()(x1)
    
    avg_pool1 = GlobalAveragePooling1D()(x11)
    max_pool1 = GlobalMaxPooling1D()(x11)
    x12 = Bidirectional(GRU(30, return_sequences=True))(x)
    
    x2 = Conv1D(30, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x12)
    avg_pool12 = GlobalAveragePooling1D()(x2)
    max_pool12 = GlobalMaxPooling1D()(x2)
    
    avg_pool2 = GlobalAveragePooling1D()(x12)
    max_pool2 = GlobalMaxPooling1D()(x12)
    
    #     x13 = Bidirectional(GRU(30, return_sequences=True))(x1)
    #     x3 = Conv1D(30, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x13)
    #     avg_pool13 = GlobalAveragePooling1D()(x3)
    #     max_pool13 = GlobalMaxPooling1D()(x2)
    #     avg_pool3 = GlobalAveragePooling1D()(x13)
    #     max_pool3 = GlobalMaxPooling1D()(x13)
    x = concatenate([avg_pool1, max_pool1,avg_pool11, max_pool11,avg_pool2, max_pool2,avg_pool12, max_pool12,capsule])
    x = Dense(6, activation = "sigmoid")(x)

    model = Model(inputs = inp, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Nadam(lr = lr, schedule_decay = lr_d), metrics = ["accuracy"])
    return model

batch_size =128
epochs = 8
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
label_cols_oof = ['toxic_oof', 'severe_toxic_oof', 'obscene_oof', 'threat_oof', 'insult_oof', 'identity_hate_oof']
pred_test = np.zeros((len(test_x), len(label_cols)))
scores = []
oof_predict = np.zeros((train_x.shape[0],6))
file_path = "best_model.hdf5"

kfidx = pickle.load(open("../input/indices.pickle", "rb"))
kfolds = 4

for trn_idx, val_idx in kfidx:
    xtr = train_x[trn_idx]
    ytr = y[trn_idx]
    xval = train_x[val_idx]
    yval = y[val_idx]
    
    model = build_model(lr = 1e-3, lr_d = 0, units = 60, dr = 0.2)

    check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 2, save_best_only = True, mode = "min")

    ra_val = RocAucEvaluation(validation_data=(xval, yval), interval = 1)
    
    early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)
    
    history = model.fit(xtr, ytr, batch_size = batch_size, epochs = epochs, validation_data = (xval, yval),verbose = 2, callbacks = [ra_val, check_point, early_stop])

    #model = load_model(file_path)
    model.load_weights(file_path)

    pred_test += model.predict(test_x, batch_size=1024,verbose = 2)

    oof_predict[val_idx] = model.predict(xval, batch_size=1024)

    cv_score = roc_auc_score(yval, oof_predict[val_idx])
    scores.append(cv_score)
    print('score: ',cv_score)

# print('Total CV score is {}'.format(np.mean(scores))) 

# oof = train_x.drop("comment_text", 1)

# oof_predict =  pd.DataFrame(oof_predict, columns=label_cols_oof)

# train_pre = pd.concat([oof,oof_predict], axis=1)

# train_pre .to_csv('yl2bigruCasp6030fast300dclean'+str(kfolds)+'_oof.csv', index=False)

# list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
# sample_submission = pd.read_csv("../input/sample_submission.csv")

# sample_submission[label_cols] = pd.DataFrame(pred_test/kfolds, columns=label_cols)
# sample_submission.to_csv('yl2bigruCasp6030fast300dclean'+'.csv', index=False)    

train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv")
# test = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv")
# embedding_path = "../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec"

print('Total CV score is {}'.format(np.mean(scores))) 

oof = train.drop("comment_text", 1)

oof_predict =  pd.DataFrame(oof_predict, columns=label_cols_oof)

train_pre = pd.concat([oof,oof_predict], axis=1)

train_pre .to_csv('yl2bigruCasp6030glove300dclean'+str(kfolds)+'_oof.csv', index=False)

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission[label_cols] = pd.DataFrame(pred_test/kfolds, columns=label_cols)
sample_submission.to_csv('yl2bigruCasp6030glove300dclean'+'.csv', index=False)    








