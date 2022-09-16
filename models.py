"""Build automations for ML training and presentation."""


from sklearn.model_selection import learning_curve, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

import pandas as pd
import numpy as np

models = {
    'Decision Tree': DecisionTreeClassifier,
    'Neural Network': MLPClassifier,
    'Boosting': GradientBoostingClassifier,
    'SVM': LinearSVC,
    'k-Nearest Neighbors': KNeighborsClassifier
}

# should keep these as hyperparams, I think
iteration_parameters = {
    'Decision Tree': 'max_depth',
    'Neural Network': 'max_iter',
    'Boosting': 'n_estimators',
    'SVM': 'max_iter',
    'k-Nearest Neighbors': 'n_neighbors'
}

random_state = 23523
scoring = 'roc_auc'

hyper_options = {
    'Decision Tree': {
        'criterion': 'entropy',
        'max_depth': [2 ** (1 + x) for x in range(5)],
        'random_state': random_state,
        'ccp_alpha': [0] + [2 ** x for x in range(-6, 3, 2)]
    },
    'Neural Network': {
        'alpha': [0] + [2 ** x for x in range(-6, 3, 2)],
        'activation': 'logistic',
        'random_state': random_state,
        'hidden_layer_sizes': [(50, 50), (100, 100), (100,), (50,), (50, 10)]
    },
    'Boosting': {
        'loss': "exponential",
        'learning_rate': 0.03,
        'n_estimators': range(30, 150, 15),
        'random_state': random_state,
        'ccp_alpha': [0] + [2 ** x for x in range(-6, 3, 2)]
    },
    'SVM': {
        'penalty': ['l1', 'l2'],
        'C': [0.01, 0.1, 0.5, 1.0, 1.5, 2.0, 5.0],
        'random_state': random_state,
        'dual': False
    },
    'k-Nearest Neighbors': {
        'n_neighbors': list(range(1, 11, 2)),
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
        # 'random_state': random_state, # R.S. unnecessary, here
        'algorithm': 'auto'
    }
}

def get_fixed_hypers(model_type):
    hp_options = hyper_options[model_type]
    fixed_hp = {
        key: hp_options[key]
        for key in hp_options
        if not isinstance(hp_options[key], list)
    }
    return fixed_hp

def get_variable_hypers(model_type):
    hp_options = hyper_options[model_type]
    variable_hp = {
        key: hp_options[key]
        for key in hp_options
        if isinstance(hp_options[key], list)
    }
    return variable_hp
    
def my_validation_curve(model_type, X, Y):
    model = models[model_type]
    
    # will use these when constructing the model initially
    fixed_hp = get_fixed_hypers(model_type)
    
    # we'll pass these in to range over for tuning
    variable_hp = get_variable_hypers(model_type)
    
    val_curves = {}
    
    for key in variable_hp:
        train_scores, valid_scores = validation_curve(
            estimator=model(**fixed_hp),
            X=X,
            y=Y,
            param_name=key,
            param_range=variable_hp[key],
            scoring=scoring
        )
        val_curves[key] = {
            'values': variable_hp[key],
            'train_scores': train_scores.mean(axis=1),
            'valid_scores': valid_scores.mean(axis=1)
        }
        
    return val_curves

def get_best_hypers_from_val_curves(val_curves):
    best_values = {}
    for key in val_curves:
        val_scores = val_curves[key]['valid_scores']
        hp_values = val_curves[key]['values']
        assert len(val_scores) == len(hp_values)
        best_value = hp_values[val_scores.argmax()]
        best_values[key] = best_value
    return best_values
        
def my_learning_curve(model_type, X, Y, all_hypers):
    model = models[model_type]
    train_sizes_abs, train_scores, valid_scores, fit_times, score_times = learning_curve(
        estimator=model(**all_hypers),
        X=X,
        y=Y,
        scoring=scoring,
        train_sizes=np.linspace(0.1, 1.0, 10),
        random_state=random_state,
        return_times=True
    )
    output = {
        'train_sizes_abs': train_sizes_abs,
        'train_scores': train_scores.mean(axis=1),
        'valid_scores': valid_scores.mean(axis=1),
        'fit_times': fit_times.mean(axis=1),
        'score_times': score_times.mean(axis=1)
    }
    return output
    
def all_curves(model_type, X, Y):
    val_curves = my_validation_curve(model_type, X, Y)
    best_variable = get_best_hypers_from_val_curves(val_curves)
    selected_hypers = get_fixed_hypers(model_type)
    selected_hypers.update(best_variable)
    learn_curves = my_learning_curve(model_type, X, Y, selected_hypers)
    output = {
        'validation_curves': val_curves,
        'learning_curves': learn_curves
    }
    return output
    