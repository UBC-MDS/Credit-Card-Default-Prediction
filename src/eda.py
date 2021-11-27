# author: Cici Du
# date: 2021-11-26

"""This script creates exploratory data visulizations to help with understanding
of the credit card default data set.

Usage: eda.py --train_visual_path=<train_visual_path> --output_dir=<output_dir>

Options:
--train_df_path=<train_visual_path>      Path including filename to training data [default: "data/processed/train_visual.csv"]
--output_dir=<output_dir>            Path to directory where the plots will be saved [default: "results/images/"]
""" 

import altair as alt
import os
import pandas as pd
from docopt import docopt


alt.data_transformers.enable('data_server') 
alt.renderers.enable('mimetype')

opt = docopt(__doc__)

def plot_cat_features(train_visual_path, output_dir):
  """
  Create and save the visuazalition of distribution of categorical features for 
  the two target variable classes.
  
  Parameters
  ----------
  train_visual_path: string
      Path including filename to training data in csv format
  output_dir: string
      Path to directory where the plots will be saved
  
  Returns
  ----------
  """

  train_df = pd.read_csv(train_visual_path)
  categorical_features = ["SEX", "EDUCATION", "MARRIAGE", 
                        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
  cat_plot = alt.Chart(train_df).mark_bar().encode(
    alt.X(alt.repeat(), type='nominal'),
    alt.Y('count()', stack=False),
    color='default payment next month',
    opacity=alt.value(0.7)
  ).properties( 
    width=200, 
    height=200 
  ).repeat( 
    categorical_features,
    columns=3
  )
  cat_plot.save(ps.path.join(output_dir, "dist_cat_feats_by_target.png"))


def plot_num_features(train_visual_path, output_dir):
  """
  Create and save the visuazalition of distribution of numeric features for 
  the two target variable classes.
  
  Parameters
  ----------
  train_visual_path: string
      Path including filename to training data in csv format
  output_dir: string
      Path to directory where the plots will be saved
  
  Returns
  ----------
  """
  train_df = pd.read_csv(train_visual_path)
  categorical_features = ["SEX", "EDUCATION", "MARRIAGE", 
                        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
  numeric_features = list(set(X_train.columns.tolist()) - 
  set(categorical_features))
  
  num_plot = alt.Chart(train_df).mark_bar().encode( 
    alt.X(alt.repeat(), type='quantitative', bin=alt.Bin(maxbins=50)), 
    alt.Y('count()', stack=False), 
    color='default payment next month',
    opacity = alt.value(0.7)
  ).properties( 
    width=200, 
    height=200 
  ).repeat( 
    numeric_features, columns=3
  ) 
  num_plot.save(os.path.join(output_dir, "dist_num_feats_by_target.png"))

def main(train_visual_path, output_dir):
  plot_cat_features(train_visual_path, output_dir)
  plot_num_features(train_visual_path, output_dir)

if __name__== "__main__":
    main(opt["--train_visual_path"], opt["--output_dir"])
