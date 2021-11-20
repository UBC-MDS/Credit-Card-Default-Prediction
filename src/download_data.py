# author: James Kim
# date: 2021-11-19

"""Downloads data from a url and saves it to a local filepat as a csv.

Usage: download_data.py --url=<url> --out_file=<out_file>

Options:
--url=<url>               URL from where to download the data (must be xls format)
--out_file=<out_file>     Path (including the file name) of where to write the file locally
"""

import os
import pandas as pd
from docopt import docopt

opt = docopt(__doc__)

def main(url, out_file):
    data = pd.read_excel(url, header=1)
    try:
        data.to_csv(out_file, index=False)
    except:
        os.makedirs(os.path.dirname(out_file))
        data.to_csv(out_file, index=False)


if __name__ == "__main__":
    main(opt["--url"], opt["--out_file"])
