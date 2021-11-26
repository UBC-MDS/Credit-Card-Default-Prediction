# author: Rohit Rawat
# date: 2021-11-26

"""This script prints out docopt args.
Usage: demo.py <arg1> [<arg4>]  --arg2=<arg2> [--arg3=<arg3>]

Options:
--path=<path>               Path to read file from
--score_file=<score_file>   Path (including filename) of where to locally save cross val score
--model_path=<model_path>   Path for the model pickle file [default: ../results/models/final_svc.pkl]
""" 

import os
import pandas as pd
from docopt import docopt
import numpy as np
import sys

from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_validate,
)

from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC
from scipy.stats import loguniform

import pickle


opt = docopt(__doc__)

def make_preprocessor(numeric_features, categorical_features):
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features),
        (StandardScaler(), numeric_features),
        )
    return preprocessor

def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation

    Parameters
    ----------
    model :
        scikit-learn model
    X_train : numpy array or pandas DataFrame
        X in the training data
    y_train :
        y in the training data

    Returns
    ----------
        pandas Series with mean scores from cross_validation
    """

    scores = cross_validate(model, X_train, y_train, **kwargs)

    mean_scores = pd.DataFrame(scores).mean()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append(mean_scores[i])

    return pd.Series(data=out_col, index=mean_scores.index)

def train_multiple_models(preprocessor, X_train, y_train, scoring_metrics):
    models = {
        "Dummy Classifier": DummyClassifier(),
        "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
        "RBF SVM": SVC(class_weight='balanced'),
        "Random Forest": RandomForestClassifier(class_weight='balanced'),
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=2000)
    }
    #Cross val-score
    results = {}
    for i in models:
        pipe_temp = make_pipeline(preprocessor, models[i])
        results[i] = mean_std_cross_val_scores(pipe_temp, X_train, y_train, scoring=scoring_metrics)
    return pd.DataFrame(results)
    
def hyperparameter_tuning(preprocessor, X_train, y_train, scoring_metrics):
    param_dists = {
        "logisticregression__C": loguniform(1e-3, 1e3),
        "logisticregression__class_weight": ['balanced', None]
    }      
    lr_pipe = make_pipeline(preprocessor, LogisticRegression(max_iter=2000))
    lr_random = RandomizedSearchCV(
        lr_pipe,
        param_dists,
        n_iter=30,
        n_jobs=-1,
        random_state=123,
        scoring=scoring_metrics,
        refit='f-1'
    )
    lr_random.fit(X_train, y_train)

    pd.DataFrame(lr_random.cv_results_)[
        ["param_logisticregression__C",
        "param_logisticregression__class_weight",
        "mean_test_accuracy",
        "mean_test_f1",
        "mean_test_recall",
        "mean_test_precision",
        "mean_test_roc_auc"]].sort_values(by='mean_test_roc_auc', ascending=False).T
    print("Best hyperparameter values: ", lr_random.best_params_)
    print("Best score: %0.3f" % (lr_random.best_score_))
    return lr_random.best_estimator_, lr_random.best_params_
    

def main(path, out_file, model_path):
    # Reading the data
    train_df = pd.read_csv(path)
    
    # Splitting between Features
    X_train, y_train = train_df.drop(columns=["default payment next month"]), train_df["default payment next month"]

    #Dividing the different types of features
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2','PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    numeric_features = ['LIMIT_BAL', 'AGE'] + X_train.columns.tolist()[11:]
    drop_features = ['ID']

    preprocessor = make_preprocessor(numeric_features, categorical_features)

    scoring_metrics = ["accuracy", "f1", "recall", "precision", "roc_auc"]
        
    results_in_df = train_multiple_models(preprocessor, X_train, y_train, scoring_metrics)
    # Output: saving the cross val score in a CSV file
    
    try:
        results_in_df.to_csv(out_file)
    except:
        os.makedirs(os.path.dirname(out_file))
        results_in_df.to_csv(out_file)

    # Model Tuning 
    # LR was decided to be the best model for this scenario
    best_model, best_params = hyperparameter_tuning(preprocessor, X_train, y_train, scoring_metrics)

    try:
        pickle.dump(best_model, open(str(model_path),"wb"))
        pickle.dump(best_params, open(str(os.path.dirname(out_file))+"/final_parameters.pkl","wb"))
    except:
        os.makedirs(os.path.dirname(model_path))
        directory = os.path.dirname(model_path)
        pickle.dump(best_model, open(model_path,"wb"))
        pickle.dump(best_params, open(str(directory)+"/final_parameters.pkl","wb"))

    

if __name__ == "__main__":
    main(opt["--path"], opt["--score_file"],opt["--model_path"])