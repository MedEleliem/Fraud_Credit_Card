import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import os
####
#creating results directory
if not os.path.isdir(r'results\before_resampling'):
    os.makedirs(r'results\before_resampling')


###############
########## DATA PREP ###########
data_path = 'creditcard.csv'
df = pd.read_csv(data_path)

########## DATA REPORTING #########
#Class Count
No_Frauds = df['Class'].value_counts()[0]
Frauds = df['Class'].value_counts()[1]


count = {"No Frauds " : int(No_Frauds), "Frauds " : int(Frauds)}
with open(r'results\before_resampling\NFvsF_before_undersampling.json', 'w') as fd:
    json.dump(count, fd)


    
# image/Class Count formatting
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style="whitegrid")
ax = sns.countplot('Class', data=df)
ax.set_xlabel('Class',fontsize = axis_fs) 
ax.set_ylabel('Count', fontsize = axis_fs)#ylabel
ax.set_title('Class Count', fontsize = title_fs)
plt.tight_layout()
plt.savefig(r'results\before_resampling\Count_before_undersampling.png',dpi=120) 
plt.close()


# image/Class Distribution formatting


ax = sns.distplot(df['Class'], color="dodgerblue", label="Compact")
ax.set_xlabel('Class',fontsize = axis_fs) 
ax.set_ylabel('Density', fontsize = axis_fs)#ylabel
ax.set_title('Class Distribution before undrsampling', fontsize = title_fs)

plt.tight_layout()
plt.savefig(r'results\before_resampling\distrib_before_undersampling.png',dpi=120) 
plt.close()

###results show that our data-set is affected with a skewed distribution, it needs to be balanced using the random undersampling techincs""" 
