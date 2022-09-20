# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 08:01:19 2022

@author: rache
"""

import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection
import seaborn as sns

from plots import plot_validation_curve
from plots import plot_learning_curve
from plots import plot_time_curve

from models import all_curves
from models import get_variable_hypers
from models import random_state

from pathlib import Path

import pickle

# --- load

X_bus = pd.read_pickle('data/business-classification/X.pkl')
Y_bus = pd.read_pickle('data/business-classification/Y.pkl')

X_wine = pd.read_pickle('data/wine/X.pkl')
Y_wine = pd.read_pickle('data/wine/Y.pkl')

# --- split 

X_bus_train, X_bus_test, Y_bus_train, Y_bus_test = sklearn.model_selection.train_test_split(
    X_bus.values, Y_bus.values,
    test_size=0.3,
    random_state=random_state
)

X_wine_train, X_wine_test, Y_wine_train, Y_wine_test = sklearn.model_selection.train_test_split(
    X_wine, Y_wine,
    test_size=0.3,
    random_state=random_state
)

training_datasets = {
    'BusClass': (X_bus_train, Y_bus_train),
    'Wine': (X_wine_train, Y_wine_train)
}

# get plots

DPI = 1200


model_types = [
    'Decision Tree',
    'Neural Network',
    'Boosting',
    'SVM',
    'k-Nearest Neighbors'
]

curve_repo = {}

for key in training_datasets:
    
    currently = pd.Timestamp('now').strftime('%Y%m%d_%H%M%S')
    dirname = f'plots/plots-{key}-{currently}'
    
    X, Y = training_datasets[key] 
    curve_repo[key] = {}
    
    for model_type in model_types:
        print(f"Getting curves for {key}'s {model_type}")
        Path(f'{dirname}/{model_type}').mkdir(parents=True, exist_ok=True)

        curves = all_curves(model_type, X, Y)
        curve_repo[key][model_type] = curves
        
        for i, variable_hyper in enumerate(get_variable_hypers(model_type)):
            val_curve = plot_validation_curve(
                curves,
                model_type=model_type,
                param_name=variable_hyper,
                dataset_name=key
            ).get_figure()
            val_curve.savefig(f'{dirname}/{model_type}/val_curve_{i}.png', dpi=DPI)
            
        learn_curve = plot_learning_curve(
            curves,
            model_type=model_type,
            dataset_name=key
        ).get_figure()
        learn_curve.savefig(f'{dirname}/{model_type}/learn_curve.png', dpi=DPI)
        
        time_curve = plot_time_curve(
            curves,
            model_type=model_type,
            dataset_name=key
        ).get_figure()
        time_curve.savefig(f'{dirname}/{model_type}/time_curve.png', dpi=DPI)
        
pickle.dump(curve_repo, 'curves.pkl')
