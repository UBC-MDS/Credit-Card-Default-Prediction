# Project Proposal

Contributors: Cici Du, James Kim, Rohit Rawat, Tianwei Wang

A data analysis project titled 'Credit Card Default Prediction' for DSCI 522 (Data Science workflows); a course in the Master of Data Science program at the University of British Columbia. The report of the analysis can be found [here](https://github.com/garhwalinauna/Credit-Card-Default-Prediction/blob/main/reports/_build/pdf/book.pdf).


## Summary

This project aims to predict the default chances of a customer based on the payment history of the customer. The data has been taken from UCI Machine Learning Repository. The default rate of customers has a direct impact on the financials of a credit card company. It is important to predict and implement processes to attenuate and adopt methods to minimize this rate. By targeting customers who are at the risk of default, the company can plan and mitigate the issue. The research question which we aim to answer through this analysis is:

## Aim

Given characteristics (gender, education, age, marriage) and payment history of a customer, is he or she likely to default on the credit card payment next month? 

## Dataset

The data set used in this project contains records of credit card customers in Taiwan sourced by I-Cheng Yeh at the Department of Information Management, Chung Hua University, Taiwan. It was downloaded from the UCI Machine Learning Repository (Dua and Graff 2019) and can be found [here](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients). Each row in the data set represents variables associated with a customer and his or her credit card payment information, including a boolean value of default. There are 30,000 observations in the data set and 23 features. There are no observations with missing values or duplicated rows in the data set.

The below explanations are provided for the features that are less intuitive.

##### BILL_AMT1 ~ BILL_AMT6
BILL_AMT1: amount of bill statement in September, 2005.   
BILL_AMT2: amount of bill statement in August, 2005   
. . .  
BILL_AMT6: amount of bill statement in April, 2005.

##### PAY_AMT1 ~ PAY_AMT6
PAY_AMT1: amount paid in September, 2005   
PAY_AMT2: amount paid in August, 2005  
. . .  
PAY_AMT6: amount paid in April, 2005.

##### PAY_0 ~ PAY_6:
PAY_0: the repayment status in September, 2005  
PAY_2: the repayment status in August, 2005  
. . .  
PAY_6: the repayment status in April, 2005. 

The values of PAY_0 ~ PAY_6 can be interpreted as:  
-1 = pay duly  
1 = payment delay for one month  
2 = payment delay for two months  
. . .  
8 = payment delay for eight months  
9 = payment delay for nine months and above.

## Process

We will begin with basic exploratory data analysis on our training dataset, identifying the data types of features, searching for missing values, scaling some of the features, and encoding categorical variables into useable features. The possible supervised learning techniques we could use are DecisionTree, Logistic Regression and SVC since we are dealing with a classification problem. We will further tune the hyperparameters of our models, and analyze feature importance as we make progress in model training. 

After choosing our final model, we will re-fit the model on the entire training data set after preprocessing and evaluate its performance on the test data set. At this point, we will look at overall accuracy as well as misclassifications (from the confusion matrix) to assess prediction performance. We also make use of Average Precision and ROC-AUC value to asses our final results as these provide us a good metric for different thresholds. We will adjust the main evaluation metric as we progress in the project.


## Usage

For reproducing the results of this repository, run the scripts in the order provided below:  

Downloading the dataset:
```
python src/download_data.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls --out_file=data/raw/data.csv
```
Cleaning and splitting the dataset:
```
python src/clean_split.py --input_file=data/raw/data.csv --test_size=0.2 --output_path=data/processed/
```
Exploratory Data Analysis:
```
python src/eda.py --train_visual_path=data/processed/train_visual.csv --output_dir=results/images/
```
Model building, training and tuning the parameters:
```
python src/model_train_tune.py --path=data/processed/train.csv --model_path=results/models/final_model.pkl --score_file=results/model_results.csv
```
Model evaluation:
```
python src/model_evaluate.py data/processed/train.csv data/processed/test.csv results/models/final_model.pkl --out_dir=results/
```

## Dependencies
The complete list of packages used can be found in the [environment file](https://github.com/UBC-MDS/Credit-Card-Default-Prediction/blob/main/environment.yaml).

The steps to using the environment are given below:

Creating an environment ```conda env create --file environment.yaml```

Activate the environment ```conda activate credit_default```

Deactivate the environment ```conda deactivate```

## License
The Credit Card Default Prediction materials here are licensed under the Creative Commons Attribution 2.5 Canada License (CC BY 2.5 CA). If re-using/re-mixing please provide attribution and link to this webpage.

## References
 
1. “Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.” 
2. The dataset can be found [here.](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
