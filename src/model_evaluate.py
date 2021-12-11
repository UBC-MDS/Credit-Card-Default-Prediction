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
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    classification_report,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score,
    average_precision_score
)

import numpy as np


opt = docopt(__doc__)

def load_model(pickle_path):
    pickle_model = open(pickle_path, "rb")
    pipe_final = pickle.load(pickle_model)
    return pipe_final

def main(train_path, test_path, model_path, out_path):
    # Reading the data
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
    except:
        print('Error in loading the train/test dataset.')
    
    # Splitting between Features
    X_train, y_train = train_df.drop(columns=["default payment next month"]), train_df["default payment next month"]
    X_test, y_test = test_df.drop(columns=["default payment next month"]), test_df["default payment next month"]
    
    try:
        final_model = load_model(model_path)
        print(final_model.fit(X_train, y_train))
    except:
        print('Error in loading the model pickle file.')
    
    # Classification Report
    classification_report_df = pd.DataFrame(
        classification_report(
            y_test, final_model.predict(X_test), target_names=["No-churn", "Churn"], output_dict=True
        )
    ).T

    classification_report_df = classification_report_df.round(decimals=3)
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)  #
    pd.plotting.table(ax, classification_report_df, loc='center')
    plt.savefig('results/images/classification_report.png', bbox_inches='tight', dpi=600)
    plt.clf()

    # Confusion Matrix on the test results
    cm = ConfusionMatrixDisplay.from_estimator(
        final_model, X_test, y_test, values_format="d", display_labels=["No default", "Default"]
        )
    plt.savefig('results/images/confusion_matrix.png')
    plt.clf()
    
    # ROC AUC curve for the test results
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
    plt.clf()

    # Precision Recall Curve
    PrecisionRecallDisplay.from_estimator(final_model, X_test, y_test)
    plt.savefig('results/images/precision_recall_curve.png')
    plt.clf()

    # Model coefficients
    categorical_columns = final_model.named_steps["columntransformer"].named_transformers_["onehotencoder"].get_feature_names_out().tolist()
    numeric_columns = final_model.named_steps["columntransformer"].named_transformers_["standardscaler"].get_feature_names_out().tolist()

    pipe_new_coeffs = final_model.named_steps["logisticregression"].coef_.tolist()
    model_coefficients = pd.DataFrame(
    data={"features": categorical_columns + numeric_columns,
          "coefficients": pipe_new_coeffs[0],
         "magnitude": np.abs(pipe_new_coeffs[0])}
    ).sort_values(by="magnitude", ascending=False).reset_index(drop=True)

    model_coeff = pd.DataFrame(model_coefficients.head(n=10)).round(decimals=3)
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)  #
    pd.plotting.table(ax, model_coeff, loc='center')
    plt.savefig('results/images/model_coefficients.png', bbox_inches='tight', dpi=600)
    plt.clf()

    # Final Scores for Model
    y_pred = final_model.predict(X_test)

    final_results = {}
    final_results['Accuracy'] = final_model.score(X_test, y_test)
    final_results['F1'] = round(f1_score(y_test, y_pred), 3)
    final_results['Recall'] = round(recall_score(y_test, y_pred), 3)
    final_results['Precision'] = round(precision_score(y_test, y_pred), 3)
    final_results['ROC AUC'] = round(roc_auc_score(y_test, final_model.decision_function(X_test)),3)
    final_results['Average Precision'] = round(average_precision_score(y_test, final_model.decision_function(X_test)),3)


    final_scores = pd.DataFrame(data=final_results,index=['Test Scores']).T.round(decimals=3)
    ax = plt.subplot(111, frame_on=False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)  #
    pd.plotting.table(ax, final_scores, loc='center')
    plt.savefig('results/images/final_scores.png', bbox_inches='tight', dpi=600)



if __name__ == "__main__":
    main(opt['<train_path>'], opt['<test_path>'], opt['<model_path>'], opt['--out_dir'])