# Credit Card Default Prediction Pipe
# author: David Wang
# date: 2021-12-03

all: results/images/classification_report.png results/images/confusion_matrix.png results/images/roc_auc_curve.png results/images/precision_recall_curve.png results/images/final_scores.png results/model_results.csv results/images/dist_num_feats_by_target.png results/images/dist_cat_feats_by_target.png results/images/dist_age_by_target.png results/images/dist_target.png

# Downloading the dataset:
data/raw/data.csv: src/download_data.py 
	python src/download_data.py --url=https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls --out_file=data/raw/data.csv

# Cleaning and splitting the dataset:
data/processed/test.csv data/processed/train.csv data/processed/train_visual.csv: src/clean_split.py data/raw/data.csv
	python src/clean_split.py --input_file=data/raw/data.csv --test_size=0.2 --output_path=data/processed/

# Exploratory Data Analysis:
results/images/dist_num_feats_by_target.png results/images/dist_cat_feats_by_target.png results/images/dist_age_by_target.png results/images/dist_target.png: src/eda.py data/processed/train_visual.csv
	python src/eda.py --train_visual_path=data/processed/train_visual.csv --output_dir=results/images/

# Model building, training and tuning the parameters:
results/models/final_model.pkl results/model_results.csv: src/model_train_tune.py data/processed/train.csv
	python src/model_train_tune.py --path=data/processed/train.csv --model_path=results/models/final_model.pkl --score_file=results/model_results.csv

# Model evaluation:
results/images/classification_report.png results/images/confusion_matrix.png results/images/roc_auc_curve.png results/images/precision_recall_curve.png results/images/final_scores.png: src/model_evaluate.py data/processed/train.csv data/processed/test.csv results/models/final_model.pkl
	python src/model_evaluate.py data/processed/train.csv data/processed/test.csv results/models/final_model.pkl --out_dir=results/

clean: 
	rm -rf data
	rm -rf results