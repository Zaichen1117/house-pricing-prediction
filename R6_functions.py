# -*- coding: utf-8 -*-
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.base import is_classifier, is_regressor
from sklearn.feature_selection import RFECV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


import matplotlib.pyplot as plt
from IPython.display import display



def check_grid_search_hyperparameters(gridCV: object, param_grid: dict) -> None:
    """Check if the best hyperparameter is on the edge of the param_grid.

    Args:
        gridCV (object): a trained sklearn.model_selection.GridSearchCV object
        param_grid (dict): dictionary of hyperparameters used in the grid search

    Returns:
        None: print warning if the best hyperparameter is on the edge of the param_grid

    Examples:
        >>> from sklearn.datasets import load_breast_cancer
        >>> from sklearn.model_selection import GridSearchCV
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> from c4_helpers import check_grid_search_hyperparameter
        >>> # load dataset
        >>> cancer = load_breast_cancer()
        >>> X, y = cancer.data, cancer.target
        >>> # prepare grid search
        >>> param_grid = {'n_estimators': [10, 20, 30],
        ...               'max_depth': [3, 5, 7]}
        >>> gridCV = GridSearchCV(RandomForestClassifier(),
        ...                       param_grid=param_grid,
        ...                       cv=3,
        ...                       scoring='roc_auc')
        >>> gridCV.fit(X, y)
        >>> # check if the best hyperparameter is on the edge of the param_grid
        >>> check_grid_search_hyperparameter(gridCV, param_grid)
        Warning: max_depth=7 is on the maximum edge of the param_grid. Please Enlarge the param_grid of this parameter.
    """

    # check if GridSearchCV has been fitted
    if not gridCV.best_params_:
        raise ValueError("The GridSearchCV object has not been fitted yet.")

    # check if param_grid is a dictionary
    if not isinstance(param_grid, dict):
        raise TypeError("param_grid should be a dictionary.")

    # Get min and max values of param_grid with numerical values only
    param_grid_num = {k: v for k, v in param_grid.items() if isinstance(v[0], (int, float))}
    param_grid_num_min = {k: min(v) for k, v in param_grid_num.items()}
    param_grid_num_max = {k: max(v) for k, v in param_grid_num.items()}
    param_grid_num_min, param_grid_num_max

    # check if the values of the grid search are on the edges of the param_grid
    for k, v in param_grid_num_min.items():
        if v == gridCV.best_params_[k]:
            print(
                f"Warning: {k}={gridCV.best_params_[k]} is on the minimum edge of the param_grid."
                " Please Enlarge the param_grid of this parameter."
            )
    for k, v in param_grid_num_max.items():
        if v == gridCV.best_params_[k]:
            print(
                f"Warning: {k}={gridCV.best_params_[k]} is on the maximum edge of the param_grid."
                " Please Enlarge the param_grid of this parameter."
            )


def construct_preprocessor(numerical_features: list, nominal_features: list) -> ColumnTransformer:
    """Construct the preprocessor to be used in sklearn pipeline
       Reference: https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html

    Args:
        numerical_features (list): list of numerical features contained in the dataset
        nominal_features (list): list of nominal features contained in the dataset

    Returns:
        ColumnTransformer: preprocessor to be used in the pipeline
    """

    numerical_transformer = Pipeline(
        steps=[
            ("num_imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    nominal_transformer = Pipeline(
        steps=[
            ("nom_imputer", SimpleImputer(strategy="most_frequent")),
            ("nominal_encoder", OneHotEncoder(handle_unknown="error", drop="first")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("nom", nominal_transformer, nominal_features),
        ]
    )

    return preprocessor

