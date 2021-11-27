# Results

## Model Results
We evaluated the model from pickle on the test dataset and we obtained comparable test scores to the validation score. We plotted the Confusion Matrix and the ROC-AUC curve corresponding to the model on the predicted labels.

![Confusion Matrix of the predictions](..results/images/confusion_matrix.png)

![ROC-AUC curve of the predictions](..results/images/roc_auc_curve.png)

## Reservations and Suggestions
Major limitation of this project is that the data was collected in 2005. Consumers’ spending behaviours and tastes must have changed since then so the results of this project should not be taken for granted and be blindly applied to the current setting. To further improve this model in the future, we suggest including more features such as income, vocation, size of the household, and debt to asset ratio. With more relevant features to base the predictions on, we should be able to predict our target class with more accuracy.