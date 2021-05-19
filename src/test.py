from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import statsmodels.api as sm
import json
import sys


if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write('\tpython src/test.py results/finalvar.pkl results/logit.pkl\n')
    sys.exit(1)

input1 = sys.argv[1]
input2 = sys.argv[2]
#####  Model Testing phase 1 (on subsample) report #####

X_test = pd.read_csv(r'results/splited_data/test_features.csv')
y_test = pd.read_csv(r'results/splited_data/test_labels.csv')
test_list = y_test.values.tolist()

with open(input1, 'rb') as fd:
    our_logit_model = pickle.load(fd)
with open(input2, 'rb') as fd:
    variables = pickle.load(fd)


X_ntest = X_test[variables]
X_ntest = sm.add_constant(X_ntest)
yhat = our_logit_model.predict(X_ntest)
prediction = list(map(round, yhat))


# confusion matrix
cm = confusion_matrix(test_list, prediction)   
# accuracy score of the model
a_s = accuracy_score(test_list, prediction)
# classification report of the model
cr = classification_report(test_list,prediction)


with open(r'results/accuracy(newsample).json', 'w') as fd:
    json.dump({'model_accuracy':a_s}, fd)
#saving confusion matrix as png    
df_cm = pd.DataFrame(cm, [0, 1])

sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) 
plt.tight_layout()
plt.savefig(r'results/confusion-matrix(newsample).png',dpi=120) 
plt.close()


with open(r'results/OnNewSample_classification_report.txt','w') as outfile : 
        outfile.write(cr)
     

#####  Model Testing phase 2 (on the whole data-set) report #####

raw_data_path = 'creditcard.csv'
df = pd.read_csv(raw_data_path)
X = df.iloc[:,0:30]
y = df['Class']

our_logit_model = pickle.load(open(r'results/logit.pkl', 'rb'))
variables = pickle.load(open(r'results/finalvar.pkl', 'rb'))

X_ntest = X[variables]
X_ntest = sm.add_constant(X_ntest)
yhat = our_logit_model.predict(X_ntest)
prediction = list(map(round, yhat))
 
# confusion matrix
cm = confusion_matrix(y, prediction)   
# accuracy score of the model
a_s = accuracy_score(y, prediction)
# classification report of the model
cr = classification_report(y, prediction)


with open(r'results/accuracy(whole_data-set).json', 'w') as fd:
    json.dump({'model_accuracy':a_s}, fd)
#saving confusion matrix as png    
df_cm = pd.DataFrame(cm, [0, 1])

sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) 
plt.tight_layout()
plt.savefig(r'results/confusion-matrix(whole_data-set).png',dpi=120) 
plt.close()


with open(r'results/Whole_data-set_classification_report.txt','w') as outfile : 
        outfile.write(cr)
     