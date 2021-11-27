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
import matplotlib.pyplot as plt

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
    print(final_model.fit(X_train, y_train))
    
    # Confusion Matrix on the test results
    cm = ConfusionMatrixDisplay.from_estimator(
        final_model, X_test, y_test, values_format="d", display_labels=["No default", "Default"]
        )
    plt.savefig('results/images/confusion_matrix.png')
    plt.clf()
    
    fpr_svc, tpr_svc, thresholds_svc = roc_curve(
        y_test, final_model.decision_function(X_test)
        )
    close_zero_svc = np.argmin(np.abs(thresholds_svc))
    plt.plot(fpr_svc, tpr_svc, label="logistic regression")
    plt.plot(
        fpr_svc[close_zero_svc],
        tpr_svc[close_zero_svc],
        "o",
        markersize=10,
        label="default threshold lr",
        c="r",
    )
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate (Recall)")
    plt.legend(loc="best")
    plt.savefig('results/images/roc_auc_curve.png')


if __name__ == "__main__":
    main(opt['<train_path>'], opt['<test_path>'], opt['<model_path>'], opt['--out_dir'])