# Executive Summary
Here we are attempt to build a classification model using various classifiers including RBF SVM, Random Forest and Logistic Regression to predict whether the customer will default on the credit card. Our chosen classifier, Logistic Regression, performed well on the test set, with the ROC AUC score of 0.768. However, as the stronger emphasis is on correctly identifying the default class, it is alarming to see relatively low scores on both f1 and recall metrics across all the classifiers tested. It is therefore recommended to further improve this model, following the suggestions that are noted in the later portion of this report.
# Introduction
## Research Question
Credit cards are now an extremely common means of transaction that most of the adult consumers possess these days. It is therefore very important for the credit card issuing companies to be able to predict and work with the possibilities of their customers not being able to make their default payments. With this in mind, our research question that we aim to answer is: given characteristics (gender, education, age, marriage) and payment history of a customer, is he or she likely to default on the credit card payment next month?
## Data 
The data set that we used was put together by I-Cheng Yeh at the Department of Information Management, Chung Hua University, in Taiwan. The data set itself was sourced from the UCI Machine Learning Repository and can be found here https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients. Each row in the data set represents variables associated with a customer and his or her credit card payment information, including a boolean value of default. There are 30,000 observations in the data set and 23 features. There are no observations with missing values or duplicated rows in the data set.
## Initial EDA
### Distribution of target variables 
We explored the distribution of the target variables and spotted class imbalance. Our training data contained only 22.3% of class 1 (default) in the target variable. We decided to balance the class during model training by setting class_weight to ‘balanced’.

```{figure} ../results/images/dist_target.png
---
name: Distribution of targets
---
Distribution of targets
```

### Distribution of numeric and categorical features by target variable 
We had 22 features and we wanted to see if any feature contributed significantly to the classification of the target variable to the extent that we could see it by plotting the distribution of each feature by the target class. We plotted the distribution of each numeric and categorical feature from the training set and colored the distribution by class (default: blue, not default: orange). We saw that the distributions below overlapped for the two classes and they looked quite similar in a lot of cases.

```{figure} ../results/images/dist_cat_feats_by_target.png
---
name: Distribution of categorical features
---
Distribution of categorical features
```


```{figure} ../results/images/dist_num_feats_by_target.png
---
name: Distribution of numerical features
---
Distribution of numerical features
```
