from keras.layers import Input, Embedding, Dense, Flatten, Dropout, concatenate
from keras.layers import BatchNormalization, SpatialDropout1D, Bidirectional
from keras.callbacks import Callback
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import LSTM, GRU
from keras.layers.advanced_activations import PReLU

# Change DEBUG to False and the kernel should take 5 hours to run on Kaggle
DEBUG = False
# DEBUG = True
WHERE = 'kaggle'
FILENO = 4
NCHUNK = 120000000#60000000
OFFSET =  184903890-115000000
VAL_RUN = False

MISSING32 = 999999999
MISSING8 = 255
PUBLIC_CUTOFF = 4032690

if WHERE=='kaggle':
    inpath = '../input/'
    pickle_path ='../input/'
    suffix = ''
    outpath = ''
    savepath = ''
    oofpath = ''
    cores = 4
elif WHERE=='gcloud':
    inpath = '../.kaggle/competitions/talkingdata-adtracking-fraud-detection/'
    pickle_path = '../data/'
    suffix = '.zip'
    outpath = '../sub/'
    oofpath = '../oof/'
    savepath = '../data/'
    cores = 7

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os

def do_count( df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_countuniq( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Counting unqiue ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_cumcount( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Cumulative count by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type,copy=False)
    return( df )
    
debug = DEBUG
if debug:
    print('*** debug parameter set: this is a test run for debugging purposes ***')

    
if VAL_RUN:
    nrows=122071522
    outpath = oofpath
else:
    nrows=184903890

nchunk=NCHUNK
# val_size=53016937
val_size=2500000
frm=nrows-OFFSET

if debug:
    frm=0
    nchunk=1000000
    val_size=100000
to=frm+nchunk
fileno = FILENO

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32',
        }

if VAL_RUN:
    print('loading train data...',frm,to)
    train_df = pd.read_pickle( pickle_path+"training.pkl.gz" )[frm:to]
    train_df['click_time'] = pd.to_datetime( train_df.click_time )
    print('loading test data...')
    if debug:
        public_cutoff = 10000
        test_df = pd.read_pickle( pickle_path+"validation.pkl.gz" )[:30000]
        test_df['click_time'] = pd.to_datetime( test_df.click_time )
        y_test = test_df['is_attributed'].values
        test_df.drop(['is_attributed'],axis=1,inplace=True)
    else:
        public_cutoff = PUBLIC_CUTOFF
        test_df = pd.read_pickle( pickle_path+"validation.pkl.gz" )
        test_df['click_time'] = pd.to_datetime( test_df.click_time )
        y_test = test_df['is_attributed'].values
        test_df.drop(['is_attributed'],axis=1,inplace=True)
else:
    print('loading train data...',frm,to)
    train_df = pd.read_csv(inpath+"train.csv", parse_dates=['click_time'], skiprows=range(1,frm), nrows=to-frm, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
    # train_df = pd.read_csv("train_day789_new.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'], parse_dates=['click_time'])


    print('loading test data...')
    if debug:
        test_df = pd.read_csv(inpath+"test.csv", nrows=100000, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    else:
        test_df = pd.read_csv(inpath+"test.csv", parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
    train_df['click_id'] = MISSING32
    train_df['click_id'] = train_df.click_id.astype('uint32')


len_train = len(train_df)
test_df['is_attributed'] = MISSING8
test_df['is_attributed'] = test_df.is_attributed.astype('uint8')
train_df=train_df.append(test_df)

del test_df
gc.collect()

print('Extracting new features...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
gc.collect()

print('Doing nextClick...')
start = time.time()
train_df['click_time'] = (train_df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
train_df['nextClick'] = (train_df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1) -train_df.click_time).astype(np.float32)
train_df['nextClick'].fillna((train_df['nextClick'].median()), inplace=True)
train_df['nextClick'] = train_df['nextClick'].astype(int)
del train_df['click_time']
print('Elapsed: {} seconds'.format(time.time() - start))
gc.collect()


print('Extracting aggregation features...')
train_df = do_countuniq( train_df, ['ip'], 'channel', 'X0', 'uint8', show_max=True ); gc.collect()
train_df = do_cumcount( train_df, ['ip', 'device', 'os'], 'app', 'X1', show_max=True ); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'day'], 'hour', 'X2', 'uint8', show_max=True ); gc.collect()
train_df = do_countuniq( train_df, ['ip'], 'app', 'X3', 'uint8', show_max=True ); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'app'], 'os', 'X4', 'uint8', show_max=True ); gc.collect()
train_df = do_countuniq( train_df, ['ip'], 'device', 'X5', 'uint16', show_max=True ); gc.collect()
train_df = do_countuniq( train_df, ['app'], 'channel', 'X6', show_max=True ); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app', 'X8', show_max=True ); gc.collect()
train_df = do_count( train_df, ['ip', 'hour'], 'ip_tcount', show_max=True ); gc.collect()
train_df = do_count( train_df, ['ip', 'app'], 'ip_app_count', show_max=True ); gc.collect()
train_df = do_count( train_df, ['ip', 'app', 'os'],'ip_app_os_count', show_max=True); gc.collect()
train_df = do_cumcount( train_df, ['ip'], 'os', 'X7', show_max=False ); gc.collect()


train_df.info()
train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

target = 'is_attributed'
predictors= ['nextClick', 'app','device','os', 'channel', 'hour',
              'ip_tcount',  'ip_app_count','ip_app_os_count',
              'X0', 'X1','X2', 'X3', 'X4', 'X5', 'X6','X7','X8']

print("label encoding....")

from sklearn.preprocessing import LabelEncoder
train_df[['app','device','os', 'channel', 'hour']].apply(LabelEncoder().fit_transform)

print('predictors',predictors)

test_df = train_df[len_train:]
val_df = train_df[(len_train-val_size):len_train]
y_val = val_df['is_attributed'] 
train_df = train_df[:(len_train-val_size)]
y_train = train_df['is_attributed'] 
train_df.drop(['click_id','ip','is_attributed'],1,inplace=True)
test_df.drop(['ip','is_attributed'],1,inplace=True)
val_df.drop(['click_id','ip','is_attributed'],1,inplace=True)
test_df.to_pickle('test.pkl.gz')

print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))
train_df.info()

print("Training...")
start_time = time.time()

print ('neural network....')

max_app = np.max([train_df['app'].max(), test_df['app'].max()])+1
max_ch = np.max([train_df['channel'].max(), test_df['channel'].max()])+1
max_dev = np.max([train_df['device'].max(), test_df['device'].max()])+1
max_os = np.max([train_df['os'].max(), test_df['os'].max()])+1
max_h = np.max([train_df['hour'].max(), test_df['hour'].max()])+1
max_nc = np.max([train_df['nextClick'].max(), test_df['nextClick'].max()])+1
max_ipc = np.max([train_df['ip_tcount'].max(), test_df['ip_tcount'].max()])+1
max_ipac = np.max([train_df['ip_app_count'].max(), test_df['ip_app_count'].max()])+1
max_ipaoc = np.max([train_df['ip_app_os_count'].max(), test_df['ip_app_os_count'].max()])+1
max_X0 = np.max([train_df['X0'].max(), test_df['X0'].max()])+1
max_X1 = np.max([train_df['X1'].max(), test_df['X1'].max()])+1
max_X2 = np.max([train_df['X2'].max(), test_df['X2'].max()])+1
max_X3 = np.max([train_df['X3'].max(), test_df['X3'].max()])+1
max_X4 = np.max([train_df['X4'].max(), test_df['X4'].max()])+1
max_X5 = np.max([train_df['X5'].max(), test_df['X5'].max()])+1
max_X6 = np.max([train_df['X6'].max(), test_df['X6'].max()])+1
max_X7 = np.max([train_df['X7'].max(), test_df['X7'].max()])+1
max_X8 = np.max([train_df['X8'].max(), test_df['X8'].max()])+1

del test_df
gc.collect()

def get_keras_data(dataset):
    X = {
        'app': np.array(dataset.app),
        'ch': np.array(dataset.channel),
        'dev': np.array(dataset.device),
        'os': np.array(dataset.os),
        'h': np.array(dataset.hour),
        'nc': np.array(dataset.nextClick),
        'ipc': np.array(dataset.ip_tcount),
        'ipac': np.array(dataset.ip_app_count),
        'ipaoc': np.array(dataset.ip_app_os_count),
        'X0': np.array(dataset.X0),
        'X1': np.array(dataset.X1),
        'X2': np.array(dataset.X2),
        'X3': np.array(dataset.X3),
        'X4': np.array(dataset.X4),
        'X5': np.array(dataset.X5),
        'X6': np.array(dataset.X6),
        'X7': np.array(dataset.X7),
        'X8': np.array(dataset.X8)
    }
    return X
train_df = get_keras_data(train_df)
val_df = get_keras_data(val_df)

emb_n = 50
dense_n = 1000
in_app = Input(shape=[1], name = 'app')
emb_app = Embedding(max_app, emb_n)(in_app)
in_ch = Input(shape=[1], name = 'ch')
emb_ch = Embedding(max_ch, emb_n)(in_ch)
in_dev = Input(shape=[1], name = 'dev')
emb_dev = Embedding(max_dev, emb_n)(in_dev)
in_os = Input(shape=[1], name = 'os')
emb_os = Embedding(max_os, emb_n)(in_os)
in_h = Input(shape=[1], name = 'h')
emb_h = Embedding(max_h, emb_n)(in_h) 
in_nc = Input(shape=[1], name = 'nc')
emb_nc = Embedding(max_nc, emb_n)(in_nc) 
in_ipc = Input(shape=[1], name = 'ipc')
emb_ipc = Embedding(max_ipc, emb_n)(in_ipc) 
in_ipac = Input(shape=[1], name = 'ipac')
emb_ipac = Embedding(max_ipac, emb_n)(in_ipac) 

in_ipaoc = Input(shape=[1], name = 'ipaoc')
emb_ipaoc = Embedding(max_ipaoc, emb_n)(in_ipaoc) 

in_X0 = Input(shape=[1], name = 'X0')
emb_X0 = Embedding(max_X0, emb_n)(in_X0) 
in_X1 = Input(shape=[1], name = 'X1')
emb_X1 = Embedding(max_X1, emb_n)(in_X1) 
in_X2 = Input(shape=[1], name = 'X2')
emb_X2 = Embedding(max_X2, emb_n)(in_X2) 
in_X3 = Input(shape=[1], name = 'X3')
emb_X3 = Embedding(max_X3, emb_n)(in_X3) 
in_X4 = Input(shape=[1], name = 'X4')
emb_X4 = Embedding(max_X4, emb_n)(in_X4) 
in_X5 = Input(shape=[1], name = 'X5')
emb_X5 = Embedding(max_X5, emb_n)(in_X5) 
in_X6 = Input(shape=[1], name = 'X6')
emb_X6 = Embedding(max_X6, emb_n)(in_X6)
in_X7 = Input(shape=[1], name = 'X7')
emb_X7 = Embedding(max_X6, emb_n)(in_X7)
in_X8 = Input(shape=[1], name = 'X8')
emb_X8 = Embedding(max_X8, emb_n)(in_X8) 
fe = concatenate([(emb_app), (emb_ch), (emb_dev), (emb_os), (emb_h), 
                 (emb_nc), (emb_ipc), (emb_ipac), (emb_ipaoc), (emb_X0), (emb_X1),(emb_X2),(emb_X3),(emb_X4), (emb_X5),(emb_X6), (emb_X7),(emb_X8)])
s_dout = SpatialDropout1D(0.2)(fe)

x = Flatten()(s_dout)
x = Dense(512,init='he_normal')(x)
x = PReLU()(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
x = Dense(64,init='he_normal')(x)
x = PReLU()(x)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)
outp = Dense(1,init='he_normal', activation='sigmoid')(x)

model = Model(inputs=[in_app,in_ch,in_dev,in_os,in_h,in_nc,in_ipc,in_ipac,in_ipaoc,in_X0,in_X1, in_X2,in_X3,in_X4,in_X5, in_X6, in_X7,in_X8], outputs=outp)

batch_size = 2048#512
epochs = 4
exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(train_df) / batch_size) * epochs
lr_init, lr_fin = 0.001, 0.0001
# lr_init, lr_fin = 0.0008, 0.00008
lr_decay = exp_decay(lr_init, lr_fin, steps)
optimizer_adam = Adam(lr=0.001, decay=lr_decay)
model.compile(loss='binary_crossentropy',optimizer=optimizer_adam,metrics=['accuracy'])

model.summary()
class_weight = {0:.0001,1:.9999} # magic
model.fit(train_df, y_train, batch_size=batch_size, class_weight=class_weight, epochs=epochs, shuffle=True, verbose=2, validation_data = [val_df, y_val])
del train_df, y_train, val_df, y_val; gc.collect()
model.save_weights('dl_support.h5')

sub = pd.DataFrame()
test_df = pd.read_pickle('test.pkl.gz')
sub['click_id'] = test_df['click_id'].astype('int')
test_df.drop(['click_id'],1,inplace=True)
test_df = get_keras_data(test_df)

print("predicting....")
sub['is_attributed'] = model.predict(test_df, batch_size=batch_size, verbose=2)
del test_df; gc.collect()
print("writing....")
sub.to_csv('18featVal25nm_1.csv',index=False)