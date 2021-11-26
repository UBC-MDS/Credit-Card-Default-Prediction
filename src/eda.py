# author: Cici Du
# date: 2021-11-26

"""This script creates exploratory data visulizations to help with understanding
of the credit card default data set.

Usage: eda.py --train_df_path=<train_df_path> --output_dir=<output_dir>

Options:
--train_df_path=<train_df_path>      Path including filename to training data [default: "data/processed/train.csv"]
--output_dir=<output_dir>            Path to directory where the plots will be saved [default: "results/"]
""" 

import altair as alt
import os
import pandas as pd

alt.data_transformers.enable('data_server') 
alt.renderers.enable('mimetype')

from docopt import docopt
opt = docopt(__doc__)

def plot_cat_features(train_df_path, output_dir):
  train_df = pd.read_csv(train_df_path)
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
  cat_plot.save(ps.path.join(output_dir, "Distribution of categorical features by target variable.png"))


def plot_num_features(train_df_path, output_dir):
  train_df = pd.read_csv(train_df_path)
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
  num_plot.save(os.path.join(output_dir, "Distribution of numeric features by target variable.png"))

def main(train_df_path, output_dir):
  plot_cat_features(train_df_path, output_dir)
  plot_num_features(train_df_path, output_dir)

if __name__== "__main__":
    main(opt["--train_df_path"], opt["--output_dir"])
