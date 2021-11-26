# author: David Wang
# date: 2021-11-25

"""
Clean, preprocess and split the raw Credit Card Default dataset, save the train and test dataset in the specified output path.

Usage: src/clean_split.py --input_file=<input_file> --test_size=<test_size> --output_path=<output_path>

Options:
--input_file=<input_file>       String: Raw data file [default: "data/raw/data.csv"]
--test_size=<test_size>         Numeric: Train test split test_size [default: 0.2]
--output_path=<output_path>     String: Directory to save train and test dataset [default: "data/processed/"]
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from docopt import docopt

opt = docopt(__doc__)

def read_data(input_file):
    # Read the raw csv data file and return a pandas DataFrame
    data = pd.read_csv(input_file)
    return data

def split(data, test_size, random_state=42):
    # Train test data split
    # default random_state = 42
    # return two pandas DataFrame train_df, test_df
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=random_state)
    return train_df, test_df

def convert_target(x):
    # Convert the orginal "default payment next month" column values
    # From a number to a meaningful string
    # 1 : Default
    # 0 : Not default
    if x == 1:
        return 'Default'
    else:
        return 'Not default'

def convert_education(x):
    # Convert the orginal "EDUCATION" column values
    # From a number to a meaningful string
    # 1 : graduate school
    # 2 : university
    # 3 : high school
    # 0, 4-6 : others
    if x == 1: 
        return 'graduate school'
    elif x == 2: 
        return 'university'
    elif x == 3: 
        return 'high school'
    else: 
        return 'others'

def convert_sex(x):
    # Convert the orginal "SEX" column values
    # From a number to a meaningful string
    # 1 : male
    # 2 : female
    if x == 1:
        return 'male'
    else:
        return 'female'

def convert_marriage(x):
    # Convert the orginal "MARRIAGE" column values
    # From a number to a meaningful string
    # 1 : married
    # 2 : single
    # 0, 3 : others
    if x == 1: 
        return 'married'
    elif x == 2: 
        return 'single'
    else: 
        return 'others'

def convert_to_category(data):
    # Convert 'default payment next month', 'SEX', 'MARRIAGE' and 'EDUCATION' columns
    # from a number to a categorical string.
    data['default payment next month'] = data['default payment next month'].apply(convert_target)
    data['SEX'] = data['SEX'].apply(convert_sex) 
    data['MARRIAGE'] = data['MARRIAGE'].apply(convert_marriage) 
    data['EDUCATION'] = data['EDUCATION'].apply(convert_education)
    return data

def main(input_file, test_size, output_path):
    
    # read the data set from the input_file, default is "data/raw/data.csv"
    data = read_data(input_file)

    # convert the test_size from str to float
    test_size = float(test_size)

    # train test split the dataset
    train_df, test_df = split(data, test_size)

    # convert "default payment next month", "SEX", "MARRIAGE", "EDUCATION" columns
    # from a number to a categorical variable to improve the readability
    # drop the "ID" column in train_df and test_df

    train_df = convert_to_category(train_df)
    train_df.drop(columns=['ID'], inplace=True)

    test_df = convert_to_category(test_df)
    test_df.drop(columns=['ID'], inplace=True)

    # The output_path for train and test dataset
    train_file = output_path + 'train.csv'
    test_file = output_path + 'test.csv'

    # Save the train and test dataset to local csv files
    # default path for the processed files is "data/processed/"
    try:
        train_df.to_csv(train_file, index=False)
    except:
        os.makedirs(os.path.dirname(train_file))
        data.to_csv(train_file, index=False)
    try:
        test_df.to_csv(test_file, index=False)
    except:
        os.makedirs(os.path.dirname(test_file))
        data.to_csv(test_file, index=False)


if __name__ == "__main__":
    main(opt["--input_file"], opt["--test_size"], opt["--output_path"])