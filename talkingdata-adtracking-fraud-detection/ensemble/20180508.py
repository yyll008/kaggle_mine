import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print("Reading the data...\n")
df1 = pd.read_csv('13featVal25nm_9795.csv')
df2 = pd.read_csv('fea11_sub_it9903lb9795.csv')
df3 = pd.read_csv('13featVal9_new9785.csv')
df4 = pd.read_csv('5kfold_0420sub_it9787.csv')
# df5 = pd.read_csv('../input/log-and-harmonic-mean-lets-go/submission_log4.csv')

models = { 'df1' : {
                    'name':'wordbatch_fm_ftrl',
                    'score':97.69,
                    'df':df1 },
           'df3' :{'name':'fea11_sub_it9903lb9795',
                    'score':97.33,
                    'df':df3 },
           'df2' :{'name':'13featVal9_new9785',
                    'score':96.33,
                    'df':df2 },    
            'df4' :{'name':'5kfold_0420sub_it9787',
                    'score':97.87,
                    'df':df4 }, 
            # 'df5' :{'name':'submission_log4',
            #         'score':1,
            #         'df':df5 }, 
         }

df1.head()         

isa_lg = 0
isa_hm = 0
isa_am=0

print("Blending...\n")
for df in models.keys() : 
    isa_lg += np.log(models[df]['df'].is_attributed)
    isa_hm += 1/(models[df]['df'].is_attributed)
    isa_am +=models[df]['df'].is_attributed

isa_lg = np.exp(isa_lg/4)
isa_hm = 1/isa_hm
isa_am=isa_am/4

print("Isa log\n")
print(isa_lg[:4])
print()
print("Isa harmo\n")
print(isa_hm[:4])


sub_am = pd.DataFrame()
sub_am['click_id'] = df1['click_id']
sub_am['is_attributed'] = isa_am
sub_am.head()


isa_fin=(isa_am+isa_hm+isa_lg)/3

sub_fin = pd.DataFrame()
sub_fin['click_id'] = df1['click_id']
sub_fin['is_attributed'] = isa_fin
print("Writing...")
#sub_log.to_csv('submission_log2.csv', index=False, float_format='%.9f')
#sub_hm.to_csv('submission_hm2.csv', index=False, float_format='%.9f')
sub_fin.to_csv('submission_final.csv', index=False, float_format='%.9f')