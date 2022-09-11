"""Plot learning and validation curves."""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from models import all_curves
from models import scoring as y_label

# output = all_curves('Decision Tree', X, Y)

def plot_validation_curve(all_curves_output, model_type, param_name):
    val_curve = all_curves_output['validation_curves'][param_name]
    val_data = pd.DataFrame(val_curve)
    plot_object = val_data.plot(
        x='values',
        y=['train_scores', 'valid_scores'],
        ylabel=y_label,
        title=f"Validation Curve for {model_type} - {param_name}"
    )
    return plot_object
    
def plot_learning_curve(all_curves_output, model_type):
    learn_curve = all_curves_output['learning_curves']
    learn_data = pd.DataFrame(learn_curve)
