import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

print("Reading the data...\n")
df1 = pd.read_csv('14fVal25nm9798.csv')
df2 = pd.read_csv('fea11_sub_it9903lb9795.csv')
df3 = pd.read_csv('13featVal9_new9785.csv')
df4 = pd.read_csv('5kfold_0420sub_it9787.csv')
df5 = pd.read_csv('wordbatch_fm_ftrl9769.csv')
df6 = pd.read_csv('sub_it9793.csv')
df7 = pd.read_csv('fea11_sub_it9792.csv')
df8 = pd.read_csv('all_sub_it8_9907_9787.csv')
df9 = pd.read_csv('sub_it9903.csv')
df10 = pd.read_csv('sub_it9826.csv')
df11 = pd.read_csv('sub_it9907.csv')
df12 = pd.read_csv('sub_it9823.csv')

models = { 'df1' : {
                    'name':'14fVal25nm9798',
                    'score':97.98,
                    'df':df1 },
           'df3' :{'name':'fea11_sub_it9903lb9795',
                    'score':97.95,
                    'df':df3 },
           'df2' :{'name':'13featVal9_new9785',
                    'score':97.85,
                    'df':df2 },    
            'df4' :{'name':'5kfold_0420sub_it9787',
                    'score':97.87,
                    'df':df4 }, 
            'df5' :{'name':'wordbatch_fm_ftrl',
                    'score':97.69,
                    'df':df5 }, 
            'df6' :{'name':'sub_it9793',
                    'score':97.93,
                    'df':df6 }, 
            'df7' :{'name':'fea11_sub_it9792',
                    'score':97.92,
                    'df':df7 }, 
            'df8' :{'name':'all_sub_it8_9907_9787',
                    'score':97.87,
                    'df':df8 }, 
            'df9' :{'name':'all_sub_it8_9907_9787',
                    'score':97.87,
                    'df':df9 }, 
            'df10' :{'name':'all_sub_it8_9907_9787',
                    'score':97.87,
                    'df':df10 }, 
            'df11' :{'name':'all_sub_it8_9907_9787',
                    'score':97.87,
                    'df':df11 }, 
            'df12' :{'name':'all_sub_it8_9907_9787',
                    'score':97.87,
                    'df':df12 }, 
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

isa_lg = np.exp(isa_lg/12)
isa_hm = 1/isa_hm
isa_am=isa_am/12

print("Isa log\n")
print(isa_lg[:12])
print()
print("Isa harmo\n")
print(isa_hm[:12])


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
sub_fin.to_csv('submission_final_12.csv', index=False, float_format='%.9f')