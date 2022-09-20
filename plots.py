"""Plot learning and validation curves."""

import pandas as pd

from models import scoring as y_label


def plot_validation_curve(all_curves_output, model_type, param_name, dataset_name):
    val_curve = all_curves_output['validation_curves'][param_name]
    val_data = pd.DataFrame(val_curve)
    plot_object = val_data.plot(
        x='values',
        y=['train_scores', 'valid_scores'],
        ylabel=y_label,
        title=f"Validation Curve for {model_type} - {param_name} - {dataset_name}"
    )
    return plot_object
    
def plot_learning_curve(all_curves_output, model_type, dataset_name):
    learn_curve = all_curves_output['learning_curves']
    learn_data = pd.DataFrame(learn_curve)
    training_set_proportion_plot = learn_data.plot(
        x='train_sizes_abs',
        y=['train_scores', 'valid_scores'],
        xlabel='# Training Records',
        ylabel=y_label,
        title=f"Learning Curve for {model_type} - {dataset_name}"
    )
    return training_set_proportion_plot
    
def plot_time_curve(all_curves_output, model_type, dataset_name):
    learn_curve = all_curves_output['learning_curves']
    learn_data = pd.DataFrame(learn_curve)
    time_plot = learn_data.plot(
        x='train_sizes_abs',
        y=['fit_times', 'score_times'],
        xlabel='# Training Records',
        ylabel='Time required (s)',
        title=f"Timing of fitting and scoring for {model_type} - {dataset_name}"
    )
    return time_plot
