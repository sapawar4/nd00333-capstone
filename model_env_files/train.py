
'''

Training script for Udacity Project 3
Sandeep Pawar
Ver 1
Date Jan 27, 2021

'''
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer

from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


from azureml.core import Workspace, Experiment


from azureml.train.hyperdrive import PrimaryMetricGoal
from azureml.train.hyperdrive import BanditPolicy
from azureml.train.hyperdrive import BayesianParameterSampling
from azureml.train.hyperdrive import HyperDriveConfig
from azureml.train.hyperdrive import choice
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.widgets import RunDetails
from azureml.train.automl import AutoMLConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core import ScriptRunConfig
from azureml.core import Workspace, Environment
import argparse
from azureml.core.run import Run
import joblib

import os



seed = 123


import warnings
warnings.filterwarnings("ignore")
run = Run.get_context()



def load_data(train_df):
    
    # Load data with all the columns from the source
    # x is teh training data
    # y is the label for the training data
    
    path = 'https://raw.githubusercontent.com/sapawar4/datasets/main/datasets/SECOM/train_lasso.csv'

    
    data = pd.read_csv(path)
    
    x = data.drop(['y','Unnamed: 0'] , axis=1)
    y = data['y']
    
    
    return x, y 

dataframe = pd.read_csv('https://raw.githubusercontent.com/sapawar4/datasets/main/datasets/SECOM/train_lasso.csv')

x, y = load_data(dataframe)

def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lrc', type=float, default=1, help="LR_C")
    parser.add_argument('--rfnest', type=int, default=100, help="rf_n_estimators")
    parser.add_argument('--rfdepth', type=int, default=5, help="rf depth")
    parser.add_argument('--rfsplit', type=int, default=2, help="rf_min_sample_split")


    args = parser.parse_args()

    run.log("LR_C:", np.float(args.lrc))
    run.log("rf_n_est:", np.float(args.rfnest))
    run.log("rf_depth:", np.float(args.rfdepth))
    run.log("rf_min_sam_split:", np.float(args.rfsplit))

    
    lr = (LogisticRegression(random_state=seed, 
                             C = args.lrc ))
    rf = (RandomForestClassifier(random_state=seed, 
                                 n_estimators = args.rfnest, 
                                 max_depth = args.rfdepth,
                                 min_samples_split = args.rfsplit))
    svc = (SVC(random_state=seed, 
               probability=True))
    gbc = (GradientBoostingClassifier(random_state=seed))
  



    stack = [('svc',svc), ('gbc',gbc), ('lr',lr), ('rf',rf)]

    voting = VotingClassifier(stack, voting='soft')


    clf = Pipeline([
        ('imputer', SimpleImputer(strategy='median')), # To impute missing values
        ('threshold', VarianceThreshold(0.01)), # Remove near-constant features
        ('scaler', StandardScaler()),
        ('classification', voting)
    ])
    clf.fit(x, y)

    score = cross_val_score(clf, X=x, y=y, cv=5, scoring = 'roc_auc')
        
    
    run.log("Mean_AUC", np.float( score.mean()))
    run.log("Std_AUC", np.float( score.std()))

    #Serialize the model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(clf, 'outputs/hyperDrive_{}_{}'.format(args.lrc,args.rfnest))

if __name__ == '__main__':
    main()
