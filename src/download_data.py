# author: James Kim
# date: 2021-11-19
# last updated on: 2021-12-03
# last updated by: David Wang

"""Download the Credit Card Default dataset from a url and saves it to a local file as a csv.

Usage: src/download_data.py --url=<url> --out_file=<out_file>

Options:
--url=<url>               URL to original data set [default: https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls]
--out_file=<out_file>     Local path and file name [default: data/raw/data.csv]
"""

import os
import pandas as pd
from docopt import docopt

opt = docopt(__doc__)

def main(url, out_file):
    data = pd.read_excel(url, header=1)

    if not os.path.isdir('data/raw/'):
        os.makedirs('data/raw/')
    if not os.path.isdir('results/images/'):
        os.makedirs('results/images/')
    if not os.path.isdir('results/models/'):
        os.makedirs('results/models/')

    try:
        data.to_csv(out_file, index=False)
    except:
        os.makedirs(os.path.dirname(out_file))
        data.to_csv(out_file, index=False)


if __name__ == "__main__":
    main(opt["--url"], opt["--out_file"])
