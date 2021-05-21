from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import statsmodels.api as sm
import json
import sys


if len(sys.argv) != 9:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write('\t python src/test.py results/finalvar.pkl results/logit.pkl results/accuracy_ns.json results/confusionmatrix_ns.png results/classification_report_ns.txt results/accuracy_wd.json results/confusionmatrix_wd.png results/classification_report_wd.txt\n')
    sys.exit(1)

input1 = sys.argv[1]
input2 = sys.argv[2]

output1 = sys.argv[3]
output2 = sys.argv[4]
output3 = sys.argv[5]
output4 = sys.argv[6]
output5 = sys.argv[7]
output6 = sys.argv[8]

#####  Model Testing phase 1 (on subsample) report #####

X_test = pd.read_csv(r'results/splited_data/test_features.csv')
y_test = pd.read_csv(r'results/splited_data/test_labels.csv')
test_list = y_test.values.tolist()

with open(input1, 'rb') as fd:
    variables = pickle.load(fd)
    
with open(input2, 'rb') as fd:
    our_logit_model = pickle.load(fd)
    


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


with open(output1, 'w') as fd:
    json.dump({'model_accuracy':a_s}, fd)
#saving confusion matrix as png    
df_cm = pd.DataFrame(cm, [0, 1])

sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) 
plt.tight_layout()
plt.savefig(output2,dpi=120) 
plt.close()


with open(output3,'w') as outfile : 
        outfile.write(cr)
     


#####  Model Testing phase 2 (on the whole data-set) report #####

raw_data_path = 'creditcard.csv'
df = pd.read_csv(raw_data_path)
X = df.iloc[:,0:30]
y = df['Class']

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


with open(output4, 'w') as fd:
    json.dump({'model_accuracy':a_s}, fd)
#saving confusion matrix as png    
df_cm = pd.DataFrame(cm, [0, 1])

sns.set(font_scale=1.4) # for label size
sns.heatmap(df_cm, annot=True, annot_kws={"size": 16}) 
plt.tight_layout()
plt.savefig(output5,dpi=120) 
plt.close()


with open(output6,'w') as outfile : 
        outfile.write(cr)
     