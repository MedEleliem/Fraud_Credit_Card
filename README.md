

Structure
--------------------

    .
    ├── .dvc <- config files of dvc
    ├── .github/workflows
        ├── cml.yaml <- docker continer action, that pull data from storage, reproduce the pipeline and share the results 
    ├── src
        ├── Resampling.py <- functions that use resample technics (under/over-sampling & SMOTE)
        ├── Stepwise_model_selection.py <- selection of the best accurate model (logit or regression) 
        ├── data_understanding.py <- scripts that generate description of the raw data
        ├── prepare.py <- auxiliary functions and classes
        ├── train.py <- training the model on the output prepare
        └── test.py <- testing the model on the new sample and the whole original data-set (using pickle files)
    ├── templates
        └── index.html <- html script for the ML Product 
    ├── static
        └── css
           └── style.css   
    ├── .dvcignore <- contains name of untracked files by dvc existing remote   
    ├── .gitignore  
    ├── README.md <- this file :)
    ├── creditcard.csv.dvc <- tracked dataset
    ├── dvc.lock  <-  restricts access to stages outputs
    ├── dvc.yaml <- contains stages that create the pipeline
    ├── params.yaml <- contains all the parameters used in the src/ scripts see more on description below
    ├── report.md
    └── requirements.txt
Pipeline        
-------------------- 

                              | creditcard.csv.dvc |
                             *************************  *******
                       ******             *                  *********
                  *****                   *                           *******
               ***                        *                                  *********
                                                                                     ****
    | understand data |                | resample data |                                 *
                                                                                         *
                                          *                                              *
                                          *                                              *
                                          *                                              *
                                                                                         *
                                  | splitting data |                                     *
                                                                                         *
                                 ***            ***                                      *
                               **                  ***                                   *
                             **                       **                                 *
                                                        **                             ***
                    | train model |                         *                        *****
                                  *****                  *                   *****
                                       ******            *             ******
                                             *****       *        *****
                                                  ***    *     ***
                                                         *
                                                    | test model |   
                                                              
##### run this cmd (after doing Preparation) to see the pipeline 
```bash
dvc dag
```
-------------------- 
# Preparation

### 1. Clone this repository
```bash
git clone https://github.com/MedEleliem/Fraud_Credit_Card
```
### 2. Get data

Download creditcard.csv

```bash
dvc pull creditcard.csv
``` 
#### You can modify the params.yaml only, it contains all the parameters used in the model production, or you can modify the scripts file in src/

### 3. Restart the pipeline
```bash
dvc repro
```
#### after doing this, push it to a new branche, differences and new plots will be displayed in the github-action bot in the new pull-request

### Also you can simply edit on github, create a pull resquest, the workflow is already automated ;) 
### e.g
![image](https://user-images.githubusercontent.com/64113527/118899364-7a6f7800-b906-11eb-96a4-097917ca385c.png)

-------------------- 
# Description

#### when you are a Data-Scientist, and you are building a ML model, for example scores and plot are necessarily for showing the power of your model, supposing that you are working with a team of data-scientists and they want to get a branch of your original project in order to test the model locally and correct some errors or doing some regularizations or any other modifications that will improve your model optimality, they want to know the result of their commitments. So, you need to regenerate txt files that contains F1 scores for example or Students statistics TEST of the coefficients in case of linear regression or maybe a printed image file saved as jpg or png for a boxplot or residuals plot or scatter-plot in the python file that train your data-set, this files needs to be updated in each commitment, you need to create a YAML file coded in GO language in Docker software in every push or pull.
#### What if Dataset is too big:
#### When we have a large CSV file or a data that contains images or sound files, evaluating the model on github will be impossible because we cannot upload it on github, a new topic is introduced is DVC, DVC is built to make ML models shareable and reproducible. It is designed to handle large files, data sets, machine learning models, and metrics as well as code. So DVC can provide us to upload files on drives platforms (Google Drive for example) and linked it to our github repositories, so when one of your team want to pull the project, he will be able to download the shared Dataset also the other python files then evaluate the model locally on his machine

--------------------
# Want to do some changes ?

### params.yaml
#### get into this file

### to choose resampling technics
#### splitting_data : 
    split-param : 1 or 2 or 3 , Undersampling or Oversampling or SMOTE
    param : [0;1]
    
### To change the train/test split percentage : 
#### splitting_data :
    split-param : 0.2

#### To change the model stepwise selection Paramaters :
##### train_model :
       model_type : "logistic" or "regression"
       elimination_criteria : "aic" or "bic"
       varchar_process : "dummy_dropfirst" 
       p-value :  0.05, I dont know if there such a elimination criteria better than this <3 
