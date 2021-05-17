import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml

#creating directory
if not os.path.isdir(r'results\splited_data'):
    os.mkdir(r'results\splited_data')


###### Spliting Stage ######


balanced_data_path = 'balanced-data.csv'
balanced_df = pd.read_csv(balanced_data_path)

X = balanced_df.iloc[:,1:30]
y = balanced_df['Class']


X = balanced_df.iloc[:,1:30]
y = balanced_df['Class']

with open("params.yaml", 'r') as fd:
    params = yaml.safe_load(fd)

splitparam = params['splitting_data']['split-param']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitparam)

# Save it

X_train.to_csv("results/splited_data/train_features.csv",index=False)
X_test.to_csv("results/splited_data/test_features.csv",index=False)
y_train.to_csv("results/splited_data/train_labels.csv",index=False)
y_test.to_csv("results/splited_data/test_labels.csv",index=False)
