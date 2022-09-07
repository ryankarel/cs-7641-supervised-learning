"""
Derive dataset for business classification.

We'll convert this raw text data for each business into a "dummified",
bag-of-words style design matrix.

Original dataset from:
    https://www.kaggle.com/datasets/charanpuvvala/company-classification

"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

# set working directory to be the location of this script
raw_data = pd.read_csv("classification-dataset-v1.csv")
text_cols = ['homepage_text', 'h1', 'h2', 'h3', 'meta_keywords', 'meta_description']
selected_text_subset = raw_data.loc[:, text_cols]

single_combined_text_column = (
    selected_text_subset
    .applymap(str)
    .apply(' '.join, axis=1)
)

dummifier = CountVectorizer(
    min_df=0.0001,
    max_df=0.7,
    max_features=1000,
    binary=True,
    stop_words='english',
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b"
)
X = dummifier.fit_transform(single_combined_text_column)

X = (
    pd.DataFrame
    .sparse
    .from_spmatrix(X, columns=dummifier.get_feature_names_out())
)
Y = pd.get_dummies(raw_data['Category'])

X.to_pickle('business-classification/X.pkl')
Y.to_pickle('business-classification/Y.pkl')
