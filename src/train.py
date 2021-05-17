import pandas as pd
import statsmodels.api as sm
from Stepwise_model_selection import forwardSelection
import pickle
import yaml

X_train = pd.read_csv(r'results\splited_data\train_features.csv')
X_test = pd.read_csv(r'results\splited_data\test_features.csv')
y_train = pd.read_csv(r'results\splited_data\train_labels.csv')
y_test = pd.read_csv(r'results\splited_data\test_labels.csv')

#setting up parameters ( read more in src/Stepwise_model_selection.py)
with open(r'params.yaml', 'r') as fd:
    params = yaml.safe_load(fd)
model_type = params['train_model']['model_type'] 
elimination_criteria = params['train_model']['elimination_criteria']
varchar_process = params['train_model']['varchar_process'] 
p_value = params['train_model']['p-value'] 


final_variables = forwardSelection(X_train, y_train, model_type = model_type,elimination_criteria = elimination_criteria, varchar_process = varchar_process, sl=p_value)[3]

X_ntrain = X_train[final_variables[1:]]
X_ntrain = sm.add_constant(X_ntrain)

log_reg = sm.Logit(y_train, X_ntrain).fit()
logit_summ = log_reg.summary()

with open(r'results\model_summary.txt', 'w') as outfile:
        outfile.write(logit_summ.as_text())
        
        
with open(r'results\logit.pkl',"wb") as f1 :
    pickle.dump(log_reg,f1)
with open(r'results\finalvar.pkl',"wb") as f2 :
    pickle.dump(final_variables[1:],f2)

f1.close()
f2.close()

