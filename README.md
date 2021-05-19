

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
    ├── .dvcignore <- contains name of untracked files by dvc existing remote   
    ├── .gitignore  
    ├── README.md <- this file :)
    ├── creditcard.csv.dvc <- tracked dataset
    ├── dvc.lock  <-  restricts access to stages outputs
    ├── dvc.yaml <- contains stage that create the pipeline
    ├── params.yaml <- contains all the parameters used in the src/ scripts
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

