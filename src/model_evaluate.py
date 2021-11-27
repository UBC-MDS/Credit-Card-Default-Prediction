# author: Rohit Rawat
# date: 2021-11-26

"""This script selects the evaluates the model performance on the test data.
Usage: model_evaluate.py <train_path> <test_path> <model_path> --out_dir=<out_dir>

Options:
<train_path>                                   Path of training dataset file
<test_path>                                    Path of test dataset file
<model_path>                                   Path of trained model file/pickle path
--out_dir=<out_dir>                            Path to the output folder
""" 

from docopt import docopt
import pandas as pd
import pickle
import os
import sys

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import roc_auc_score

import numpy as np


opt = docopt(__doc__)

def load_model(pickle_path):
    pickle_model = open(pickle_path, "rb")
    pipe_final = pickle.load(pickle_model)
    return pipe_final

def main(train_path, test_path, model_path, out_path):
    # Reading the data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Splitting between Features
    X_train, y_train = train_df.drop(columns=["default payment next month"]), train_df["default payment next month"]
    X_test, y_test = test_df.drop(columns=["default payment next month"]), test_df["default payment next month"]
    
    final_model = load_model(model_path)
    print('all good')
    print(final_model.fit(X_train, y_train))
    
    ## Need to add visualization.

if __name__ == "__main__":
    main(opt['<train_path>'], opt['<test_path>'], opt['<model_path>'], opt['--out_dir'])