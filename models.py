"""Build automations for ML training and presentation."""


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

models = {
    'Decision Tree': DecisionTreeClassifier,
    'Neural Network': MLPClassifier,
    'Boosting': GradientBoostingClassifier,
    'Support Vector Machine': LinearSVC,
    'k-Nearest Neighbors': KNeighborsClassifier
}

# should keep these as hyperparams, I think
iteration_parameters = {
    'Decision Tree': 'max_depth',
    'Neural Network': 'max_iter',
    'Boosting': 'n_estimators',
    'Support Vector Machine': 'max_iter',
    'k-Nearest Neighbors': 'n_neighbors'
}

random_state = 23523

hyper_options = {
    'Decision Tree': {
        'criterion': ['entropy'],
        'max_depth': [2 ** (1 + x) for x in range(5)],
        'random_state': [random_state],
        'ccp_alpha': [0] + [2 ** x for x in range(-3, 2)]
    }
}


def cross_validate(model_type, X, Y, cv=3):
    model = models[model_type]
    hyper_grid = hyper_options[model_type]
    grid_search = GridSearchCV(
        estimator=model(),
        param_grid=hyper_grid,
        scoring='roc_auc',
        cv=cv,
        return_train_score=True
    )
    grid_search.fit(X, Y)
    pd.DataFrame(grid_search.cv_results_)
    