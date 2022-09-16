# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 07:13:06 2022

@author: rache
"""

import pandas as pd
import sklearn

from plots import plot_validation_curve
from plots import plot_learning_curve
from plots import plot_time_curve

from models import all_curves
from models import get_variable_hypers

X = pd.read_pickle('data/wine/X.pkl')
Y = pd.read_pickle('data/wine/Y.pkl')

# X = pd.read_pickle('data/business-classification/X.pkl')
# Y = pd.read_pickle('data/business-classification/Y.pkl')

output = all_curves('Decision Tree', X, Y)

model_type = 'k-Nearest Neighbors'

curves = all_curves(model_type, X, Y)

for variable_hyper in get_variable_hypers(model_type):
    plot_validation_curve(
        curves,
        model_type=model_type,
        param_name=variable_hyper
    )
plot_learning_curve(
    curves,
    model_type=model_type
)
plot_time_curve(
    curves,
    model_type=model_type
)