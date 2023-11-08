# -*- coding: utf-8 -*-
from itertools import combinations, product,compress
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Union
from IPython.display import display
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE, r_regression, SelectKBest
from sklearn.svm import SVR
from sklearn.metrics import (
    explained_variance_score,
    mean_squared_error,
    mean_absolute_error,
    max_error,
)


def chain_snap(data, fn=lambda x: x.shape, msg=None):
    r"""Print things in method chaining, leaving the dataframe untouched.
    Parameters
    ----------
    data: pandas.DataFrame
        the initial data frame for which the functions will be applied to in the pipe
    fn: lambda
        function that takes a pandas.DataFrame and that creates output to be printed
    msg: str or None
        optional message to be printed above output of the function
    Examples
    --------
    >>> from neuropy.utils import chain_snap
    >>> import pandas as pd
    >>> df = pd.DataFrame({'letter': ['a', 'b', 'c', 'c', 'd', 'c', 'a'],
    ...                    'number': [5,4,6,3,8,1,5]})
    >>> df = df.pipe(chain_snap, msg='Shape of the dataframe:')
    >>> df = df.pipe(chain_snap,
    ...              fn = lambda df: df['letter'].value_counts(),
    ...              msg="Frequency of letters:")
    >>> df = df.pipe(chain_snap,
    ...              fn = lambda df: df.loc[df['letter']=='c'],
    ...              msg="Dataframe where letter is c:")
    """
    if msg:
        print(msg + ": " + str(fn(data)))
    else:
        print(fn(data))
    return data


def standardise_data(
    X_train: pd.DataFrame, X_test: pd.DataFrame, quantitative_features: list
) -> pd.DataFrame:
    """Normalize numerical features using StandardScaler

    Args:
        X_train (pd.DataFrame): The train set coming from the train_test_split function
        X_test (pd.DataFrame): The test set coming from the train_test_split function
        quantitative_list (List): List of numerical features

    Returns:
        pd.DataFrame: The train and test set with the numerical features normalized
    """

    # initialize scaler
    scaler = StandardScaler()
    # fit scaler on train data
    _ = scaler.fit(X_train[quantitative_features])
    # transform train and test data
    X_train_scaled = pd.DataFrame(
        data=scaler.transform(X_train[quantitative_features]),
        columns=[x + "_scaled" for x in quantitative_features],
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        data=scaler.transform(X_test[quantitative_features]),
        columns=[x + "_scaled" for x in quantitative_features],
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled


def encode_categorical_features(X_train, X_test, categorical_features):
    """Encode categorical features using sklearn OneHotEncoder
    Args:
        X_train (pd.DataFrame): The train set coming from the train_test_split function
        X_test (pd.DataFrame): The test set coming from the train_test_split function
        categorical_features (list): list of categorical features

    Returns:
        X_train_cat (pd.DataFrame): Train data with encoded categorical features
        X_test_cat (pd.DataFrame): Test data with encoded categorical features
    """

    # initialize encoder
    encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    # fit encoder on train data
    _ = encoder.fit(X_train[categorical_features])

    # transform train and test data
    X_train_cat = pd.DataFrame(
        data=encoder.transform(X_train[categorical_features]),
        columns=encoder.get_feature_names_out(categorical_features),
        index=X_train.index,
    )

    X_test_cat = pd.DataFrame(
        data=encoder.transform(X_test[categorical_features]),
        columns=encoder.get_feature_names_out(categorical_features),
        index=X_test.index,
    )

    return X_train_cat, X_test_cat
