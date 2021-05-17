"""This file contains differents types of resampling techincs (Under Sampling, Over Sampling, SMOTE), to use this function run final_func and choose one of those parameters 1,2,3
    for under,over,smote resampling, in addition to input variables & output variables choosen from the dataset, and percentage for classing (only for under&over sampling"""

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import os
import yaml
import json
import seaborn as sns
import matplotlib.pyplot as plt
#######
#### Creating directory for results
if not os.path.isdir(r'results\after_resampling'):
    os.makedirs(r'results\after_resampling')
    

def under_sampler_func(inputt, output, percentage):
  #input must be a single/multiple columns in pandas dataset, output should be just one column
  undersample = RandomUnderSampler(sampling_strategy=percentage)
  X_under, y_under = undersample.fit_resample(inputt, output)
  dt = [X_under, y_under]
  dfn = pd.concat(dt, axis=1)
  return dfn

  
def over_sampler_func(inputt, output, percentage):
  #input must be a single/multiple columns in pandas dataset, output should be just one column"""
  oversample = RandomOverSampler(sampling_strategy=percentage)
  X_over, y_over = oversample.fit_resample(inputt, output)
  dt = [X_over, y_over]
  dfn = pd.concat(dt, axis=1)
  return dfn

  
def SMOTE_sampler_func(inputt, output, percentage):
  smote = SMOTE()
  Xs, ys = smote.fit_resample(inputt, output)
  dt = [Xs, ys]
  dfn = pd.concat(dt, axis=1)
  return dfn


def final_func(param,inputt, output, percentage):
  if param==1:
    return under_sampler_func(inputt, output, percentage)
  elif param==2:
    return over_sampler_func(inputt, output, percentage)
  elif param==3:
    return SMOTE_sampler_func(inputt, output, percentage)

    
raw_data_path = 'creditcard.csv'
df = pd.read_csv(raw_data_path)

x = df.iloc[:,0:30]
y = df.iloc[:,30:31]

###### instantiating the random undersampler ######
### calling for parameters from params.yaml 
with open(r'params.yaml', 'r') as fd:
    params = yaml.safe_load(fd)

param = params['resample_data']['param'] #parameters 1,2,3 for under,over,smote resampling
percentage = params['resample_data']['percentage'] #read imblearn documentary
undersampling_res = final_func(param ,x, y,percentage)
undersampling_res.to_csv(r'balanced-data.csv', index=False)
balanced_df = pd.read_csv(r'balanced-data.csv')

X = balanced_df.iloc[:,1:30]
y = balanced_df['Class']

No_Frauds_n = y.value_counts()[0]
Frauds_n = y.value_counts()[1]

with open(r'results\after_resampling\NFvsF_after_undersampling.json', 'w') as outfile:
    json.dump({ "No Frauds " : int(No_Frauds_n), "Frauds " : int(Frauds_n)}, outfile)

# image/Class Distribution formatting
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")

ax = sns.distplot(y, color="dodgerblue", label="Compact")
ax.set_xlabel('Class',fontsize = axis_fs) 
ax.set_ylabel('Density', fontsize = axis_fs)#ylabel
ax.set_title('Class Distribution after undersampling', fontsize = title_fs)

plt.tight_layout()
plt.savefig(r'results\after_resampling\Distrib_after_undersampling.png',dpi=120) 
plt.close()

###### Correlation matrices comparaison ######
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))
corr = df.corr(method='spearman')
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14)


sub_sample_corr = balanced_df.corr(method='spearman')
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
ax2.set_title('SubSample Correlation Matrix \n (use for reference)', fontsize=14)

plt.tight_layout()
plt.savefig(r'results\after_resampling\Corr_matrices_comp.png',dpi=120) 
plt.close()     


