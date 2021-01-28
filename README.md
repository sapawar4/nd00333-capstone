
# Capstone Project
## Sandeep Pawar

In this final project, I chose a high-demensional real-world messy data to showcase how Azure ML can be used to operationalize a machine learning pipeline. The goal of this project is to predict if a machine will pass or fail based on number of measurements. Many different preprocessing techniques and algorithms are explored and final model was chosen between a optimized model and an AutoML model. The model optimized with AzureML HyperDrive was deployed in service for real-time inferencing. 
 
## Architecture 

![](https://docs.microsoft.com/en-us/azure/architecture/browse/thumbs/information-discovery-with-deep-learning-and-nlp.png)

As shown in the architecture diagram above, data will be retrieved from the data source and machine learning model will be built using Azure ML Service. Remote compute cluster is used to train large number of models and the final model is deployed in service using ACI/AKS inference cluster. The end-user can rerieve it via web service API. 

## Dataset

This data is from [UCI Machine Learning repository](https://archive.ics.uci.edu/ml/datasets/SECOM) about a complex semiconductor manufacturing process. The process is monitored by logging signals from 590 different sensors. The columns are anonymized thus it's not possible to know what the columns mean. The labels represent a simple pass/fail yield for in house line testing. 

There are 590 columns total. The target label is -1 = Pass and 1 = Fail. 

This is an imbalanced dataset as ~93% of the labels belong to pass and only 7% are fail. Also, most of the columns have missing values. Some columns have constant values. 

### Observations and modeling strategy:

1. All features are numerical
2. Majority of the features have null values => Imputation will be needed
3. Features have different scales => scaling will be required
4. 122 columns have less than 3 distinct values. This means either they have near constant variance or they might be categorical.  
5. Imbalanced dataset


Keeping this in mind, we will have to build a pipeline that will include:
1. Standardization
2. Imputation
3. Variance Threshold to remove near-constant values as they will have less predictive power
4. Since number of features are very large, we will need to do dimentionality reduction. While there are many different ways, to keep things simple, I will try two approaches:
    - Feature selection by Lasso
    - By using PCA
5. Typically, use F1 or AUC as the metric since the labels are imbalanced. Also the classifier must be able to handle 'class_weight'.
6. We will try three linear algorithms (Logistic, SVC, Ridge) and two tree-based (Random Forest & lightgbm), stacking ensemble and voting ensemble. All algorithms will be based on CV to measure its 'generalizability'.
7. 5 fold cross-validation is used and mean and standard deviation in the cross-folds are computed and logged.
7. Finally, the final model for hyperparameter tuning will be based on:
    - Simple
    - Parsimonious
    - Easy to maintain, debug and interpret
    - 

### Access
- For hyperdrive, the data is access from local *.csv file
- For AutoML, the data is registered to the datastore so remote compute can be used

## Automated ML
- Since it's an imbalanced dataset, `Weighted_AUC` is used the primary metric
- Remote compute `DS2V2` was used for training
- 5 fold cross-validation was used to mitigate overfitting
- Experiment was set to timeout at 20 minutes

### Configuration

AutoML configuration is described below: 

| Configuration & Description | Value | 
|--|--|
| *experiment_timeout_minutes* - Max duration of time in min the experiment should be run | 20 |
| *task* - Training task if regression or classification | Classification |
| *primary_metric* - Metric to optimize, AUC because of imbalanced data | 'AUC_weighted' |
| *n_cross_validations* - Cross validation folds for training to prevent overfitting | 5' |
| *training_data* - dataset used for training. Registered dataset | amlds |
| *compute_target* - Remote compute cluster to use | Standard_DS2_V2 |
| *label_column_name* - target label to use | 'y' |



### Results

The best AutoML model was a Voting classifier with  Weighted_AUC = 77.58%. The votingclassifier includes several different types of classifiers and predictions are made by averaging the probabilities from the individual classifiers.  

![](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier_files/stackingclassification_overview.png)
[Reference](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/)

The Voting Classifier had following 8 models in it:

| Classifier | Parameters 
|--|--|
| RandomForestClassifier | MinMax, max_depth = none, min_sample_leaf = 0.01, n_estimators = 25 |
| ExtraTreesClassifier | Robust, max_depth = none, min_sample_leaf = 0.01, n_estimators = 25 |
| ExtraTreesClassifier | MinMax, max_depth = none, min_sample_leaf = 0.01, n_estimators = 10 |
| ExtraTreesClassifier | StandardScaler, max_depth = none, min_sample_leaf = 0.01, n_estimators = 10 |
| RandomForestClassifier | MinMax, max_depth = none, min_sample_leaf = 0.01, n_estimators = 10 | RandomForestClassifier | MinMax, max_depth = none, min_sample_split = 0.10, n_estimators = 10 
| LighGBM | MaxAbs, max_depth = -1, n_estimators=100, min_child_samples=20
| ExtraTreesClassifier | min_samples_split=0.056, max_features=0.3,n_estimators=25

Below screenshot shows how different models performed. The best models were Voting Ensemble, Stacking Ensemble and ExtremeRandomtrees. Voting classifier did significantly better than the rest.

![](https://raw.githubusercontent.com/sapawar4/nd00333-capstone/master/starter_file/images/automl.JPG)
Below screenshot shows how the RunDetails widget in the notebook and how the models performed.
![](https://raw.githubusercontent.com/sapawar4/nd00333-capstone/master/starter_file/images/automl2.JPG)

Metric information about the final model from AutoML. AUC is area under the ROC curve. For the best model, ROC = 77.45%


![](https://raw.githubusercontent.com/sapawar4/nd00333-capstone/master/starter_file/images/automl3.JPG)


## Hyperparameter Tuning with HyperDrive

- Number of different individual algorithms such as SVC, LogisticRegression, Lasso, Ridge, RandomForest, GradientBoostingClassifier were used. Finally ensembling using voting and stacking were also trained. 
- For preprocessing, all the columns were
	- Imputed
	- Scaled using Standard Scaler
	- Variance threshold was used to remove any features with constant values
	- Lasso regression & PCA were used to evaluate dimentionality reduction
- Lasso regression reduced the number of features from 590 to 207 and improved the AUC score considerably
- Voting Classifier showed highest AUC ~ 75% and was chosen for tuning
- 4 parameters were tuned:
	- Penalty of Logistic Regression, C
	- No of estimators in Random Forest
	- Depth of tree in Random Forest
	- Min sample split in Random Forest
- Bayesian optimization was used considering large number of parameters. Bayesian optimization typically converges faster and often gives better results
- Bayesian optimization does not acceept Early stopping policy, hence max time of 20 was used
- 
### Results
Final model was Voting Classifier with:
	- C: 5
	- Random forest n_estimators = 100
	- Random Forest max_depth = 7
	- Random forest min_sample_split = 3

With these parameters the mean AUC = 77%. Untuned voting classifier was 75%
RunDetails widgets showing the progress of the hyperdrive with metric for each run. 

![](https://raw.githubusercontent.com/sapawar4/nd00333-capstone/master/starter_file/images/hpo.JPG)Below plot shows how the hypertuning progresses. Bayesian optimization progressively gets better. The best model was found at run # 31. Parallel coordinate plots shows which parameters led to high scores . In general it doesnt look like any one particular parameter always performed the best, its the combination of different parameters. 


![](https://raw.githubusercontent.com/sapawar4/nd00333-capstone/master/starter_file/images/hpo2.JPG)

The difference between VotingClassifier from HyperDrive and AutoML is that, HyperDrive models use the same preprocessing steps but are different types of algorithms (tree + linear), whereas AutoML models are all tree-based and have different preprocessing steps (Standard scaling, Max Abs Scaling etc.)



## Model Deployment
As the difference in performance between hyperdrive model and automl model wasn't significant, hyperdrive model was chosen for deployment for practice. The final model was deployed in service using Azure Container Instance and was tested to ensure its healthy active status. 

![](https://raw.githubusercontent.com/sapawar4/nd00333-capstone/master/starter_file/images/service.JPG)
![](https://raw.githubusercontent.com/sapawar4/nd00333-capstone/master/starter_file/images/service%202.JPG)
Service was deleted after testing.



## Screen Recording
Youtube: https://youtu.be/hhXCZMxxNSY

## Standout Suggestions
1. Try undersampling, oversampling, SMOTE etc
2. Different imputation techniques



