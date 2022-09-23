# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 08:01:19 2022

@author: rache
"""

import pandas as pd
import numpy as np
import sklearn
import sklearn.model_selection
from sklearn.metrics import roc_auc_score
import seaborn as sns

from plots import plot_validation_curve
from plots import plot_learning_curve
from plots import plot_time_curve

from models import all_curves
from models import get_variable_hypers
from models import random_state
from models import get_best_hypers_from_val_curves
from models import get_fixed_hypers
from models import models

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

holdout_datasets = {
    'BusClass': (X_bus_test, Y_bus_test),
    'Wine': (X_wine_test, Y_wine_test)
}

# get plots

DPI = 1200


model_types = [
    'k-Nearest Neighbors',
    'Decision Tree',
    'Neural Network',
    'SVM',
    'Boosting'
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

print(curve_repo)

# --- holdout test

holdout_performance = {}

for key in ['Wine', 'BusClass']:

    X_train, Y_train = training_datasets[key]
    X_test, Y_test = holdout_datasets[key]
    holdout_performance[key] = {}

    for model_type in model_types:
        curves = curve_repo[key][model_type]
        best_variable = get_best_hypers_from_val_curves(curves['validation_curves'])
        selected_hypers = get_fixed_hypers(model_type)
        selected_hypers.update(best_variable)
        model = models[model_type](**selected_hypers)
        model.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        score = roc_auc_score(Y_test, Y_pred, multi_class="ovo")
        print(f'{key} - {model_type} score: {score:.3f}')
        
        holdout_performance[key][model_type] = score
        
