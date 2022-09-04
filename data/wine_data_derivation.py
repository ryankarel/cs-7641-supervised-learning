"""
Derive dataset for wine quality regression.

Original dataset from:
    http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/

"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

# set working directory to be the location of this script
raw_red_data = pd.read_csv("winequality-red.csv", sep=";").assign(red=1)
raw_white_data = pd.read_csv("winequality-white.csv", sep=";").assign(red=0)

combined = pd.concat([raw_red_data, raw_white_data])

X = combined.drop('quality', axis=1)
Y = combined['quality']

X.to_pickle('wine/X.pkl')
Y.to_pickle('wine/Y.pkl')
