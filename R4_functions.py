# -*- coding: utf-8 -*-
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import (
    r2_score,
    explained_variance_score,
    mean_squared_error,
    mean_absolute_error,
    max_error,
)
from IPython.display import display
from statsmodels.graphics.gofplots import ProbPlot

# standardize the numeric features
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

#One hot encoding to create dummies for categorical features
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
    encoder = OneHotEncoder(drop='first', handle_unknown="ignore", sparse_output=False)
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

# function to calculate the VIF score for regression models
def get_variance_inflation_factors(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the variance inflation factors (VIF) for the features in the dataframe X
    :param X: dataframe contraining the features
    :return: dataframe with the variance inflation factors
    """

    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns

    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    return vif_data


# generate diagnostic plots for linear regression models
def diagnostic_plots(X, y, model_fit=None):
    """
    Generates diagnostic plots for evaluating a linear regression model.

    Parameters:
    X (DataFrame or array-like): Independent variables for the regression model.
    y (Series or array-like): Dependent variable for the regression model.
    model_fit (statsmodels OLS object, optional): Pre-fitted linear regression model. 
        If None, the function fits a new model using X and y.

    Plots generated (similar to autoplot from R ggfortify package)):
    1. Residuals vs Fitted
    2. Normal Q-Q
    3. Scale-Location
    4. Cook's Distance
    5. Residuals vs Leverage
    6. Cook's Distance vs Leverage
    """
    

    if not model_fit:
        model_fit = sm.OLS(y, sm.add_constant(X)).fit()

    # create dataframe from X and y for easier plot handling
    dataframe = pd.concat([X, y], axis=1)

    # model values
    model_fitted_y = model_fit.fittedvalues
    # model residuals
    model_residuals = model_fit.resid
    # normalized residuals
    model_norm_residuals = model_fit.get_influence().resid_studentized_internal
    # absolute squared normalized residuals
    model_norm_residuals_abs_sqrt = np.sqrt(np.abs(model_norm_residuals))
    # absolute residuals
    model_abs_resid = np.abs(model_residuals)
    # leverage, from statsmodels internals
    model_leverage = model_fit.get_influence().hat_matrix_diag
    # cook's distance, from statsmodels internals
    model_cooks = model_fit.get_influence().cooks_distance[0]

    # Setup for 2x3 grid layout
    fig, axs = plt.subplots(2, 3, figsize=(20, 12))

    # Dark grey color for points
    point_color = '#404040'

    # Residuals vs Fitted with annotation for the 3 largest residuals
    sns.residplot(x=model_fitted_y, y=model_residuals, lowess=True, ax=axs[0, 0], line_kws={'color': 'blue', 'lw': 1}, scatter_kws={'alpha': 0.5, 'color': point_color})
    axs[0, 0].set_title('Residuals vs Fitted')
    axs[0, 0].set_xlabel('Fitted values')
    axs[0, 0].set_ylabel('Residuals')
    # Annotate the 3 largest residuals
    top3 = np.argsort(np.abs(model_residuals))[-3:]
    for i in top3:
        axs[0, 0].annotate(i, xy=(model_fitted_y[i], model_residuals[i]), color='red')

    # Normal Q-Q with annotation for the 3 largest normalized residuals
    QQ = ProbPlot(model_norm_residuals)
    QQ.qqplot(line='45', ax=axs[0, 1], alpha=0.5, marker='o', markerfacecolor=point_color, markeredgecolor=point_color, lw=1)
    # axs[0, 1].get_lines()[0].set_color('blue')
    # axs[0, 1].get_lines()[0].set_alpha(0.5)
    axs[0, 1].set_title('Normal Q-Q')
    axs[0, 1].set_xlabel('Theoretical Quantiles')
    axs[0, 1].set_ylabel('Standardized Residuals')
    line = axs[0, 1].get_lines()[1]  # Get the reference line
    line.set_color('gray')  # Set the color of the line to blue
    line.set_linestyle('--')  # Ensure the line is solid
    # Annotate the 3 largest normalized residuals
    for i in top3:
        axs[0, 1].annotate(i, 
                            xy=(QQ.theoretical_quantiles[model_norm_residuals.argsort().tolist().index(i)], 
                                model_norm_residuals[i]), 
                            color='red')

    # Scale-Location with annotation for the 3 largest standardized residuals
    axs[0, 2].scatter(model_fitted_y, model_norm_residuals_abs_sqrt, alpha=0.5, color=point_color)
    sns.regplot(x=model_fitted_y, y=model_norm_residuals_abs_sqrt, scatter=False, lowess=True, ax=axs[0, 2], line_kws={'color': 'blue', 'lw': 1}, scatter_kws={'alpha': 0.5, 'color': point_color})
    axs[0, 2].set_title('Scale-Location')
    axs[0, 2].set_xlabel('Fitted values')
    axs[0, 2].set_ylabel('$\sqrt{|Standardized Residuals|}$')
    # Annotate the 3 largest sqrt standardized residuals
    for i in top3:
        axs[0, 2].annotate(i, xy=(model_fitted_y[i], model_norm_residuals_abs_sqrt[i]), color='red')

    # Cook's Distance plot as bar plot with annotation for the 3 largest Cook's distances
    axs[1, 0].bar(range(len(model_cooks)), model_cooks, color=point_color)
    axs[1, 0].set_title("Cook's distance")
    axs[1, 0].set_xlabel('Observation Number')
    axs[1, 0].set_ylabel("Cook's Distance")
    # Annotate the 3 largest Cook's distance
    for i in top3:
        axs[1, 0].annotate(i, xy=(i, model_cooks[i]), color='red')

    # Residuals vs Leverage with annotation for the 3 points with the highest leverage
    sns.scatterplot(x=model_leverage, y=model_norm_residuals, ax=axs[1, 1], color=point_color, alpha=0.5)
    axs[1, 1].set_title('Residuals vs Leverage')
    axs[1, 1].set_xlabel('Leverage')
    axs[1, 1].set_ylabel('Standardized Residuals')
    # Annotate the 3 highest leverage points
    for i in top3:
        axs[1, 1].annotate(i, xy=(model_leverage[i], model_norm_residuals[i]), color='red')

    # Cook's Distance vs Leverage with annotation for the 3 largest Cook's distances
    sns.scatterplot(x=model_leverage, y=model_cooks, ax=axs[1, 2], color=point_color, alpha=0.5)
    axs[1, 2].set_title("Cook's dist vs Leverage")
    axs[1, 2].set_xlabel('Leverage')
    axs[1, 2].set_ylabel("Cook's Distance")
    # Annotate the 3 largest Cook's distance
    for i in top3:
        axs[1, 2].annotate(i, xy=(model_leverage[i], model_cooks[i]), color='red')

    plt.suptitle('Model Diagnostics Plots', size=16)
    plt.tight_layout()
    plt.show()
    


# function to summarize the performance metrics of a linear regression model
def regression_report(
    y_true: np.ndarray, y_pred: np.ndarray, label: str = "Score", show_description: str = True
) -> pd.DataFrame:
    """Generate a report for a regression model.

    Args:
        y_true (np.ndarray): the true labels of the data
        y_pred (np.ndarray): the predicted labels of the data (must be binary)
        label (str, optional): tailor score name. useful if you want to compare multiple models. Defaults to 'Score'.
        show_description (str, optional): Provide a description of each metric to the report. Defaults to True.

    Returns:
        pd.DataFrame: a dataframe containing the classification report

    Example:
        >>> import numpy as np
        >>> from r4_helpers import regression_report
        >>> y_true = np.array([1,   0.1, 0.5, 6, -2, 0, 1.1, -0.5, 12])
        >>> y_pred = np.array([0.5, 0.3, 0.2, 3, -1, 1, 1.5, -1, 1])
        >>> regression_report(y_true, y_pred)
    """

    dict_clf = {
        "r-squared": [
            r2_score(y_true, y_pred),
            "Coefficient of Determination",
        ],
        "explained variance": [
            explained_variance_score(y_true, y_pred),
            "Explained Variance Score",
        ],
        "RMSE": [
            mean_squared_error(y_true, y_pred, squared=False),
            "Root Mean Squared Error (RMSE)",
        ],
        "MAE": [
            mean_absolute_error(y_true, y_pred),
            "Mean Absolute Error (MAE)",
        ],
        "max error": [
            max_error(y_true, y_pred),
            "The maximum residual error",
        ],
    }

    # Convert Dictionary to DataFrame
    df_report = pd.DataFrame.from_dict(
        dict_clf, orient="index", columns=[label, "Description"]
    ).round(3)

    if show_description is False:
        df_report.drop(columns=["Description"], inplace=True)

    return df_report

# function to generate a report for both in-sample and out-of-sample predictions
def regression_report_in_sample_out_of_sample(
    y_train: pd.Series, y_fit: pd.Series, y_test: pd.Series, y_pred: pd.Series, model_name: str
) -> pd.DataFrame:
    """Wrapper function around regression_report to generate in-sample and out-of-sample reports.

    Args:
        y_train (pd.Series): the targer variable in the training set
        y_fit (pd.Series): the fitted values in the training set
        y_test (pd.Series): the target variable in the validation set or test set
        y_pred (pd.Series): the predicted values in the validation set or test set
        model_name (str): the name of the model

    Returns:
        pd.DataFrame: dataframe containing the regression report for both in-sample and out-of-sample predictions and their difference in percentage
    """

    # Return regression report for both in-sample (train) and out-of-sample (validation) predictions
    return (
        pd.merge(
            left=regression_report(
                y_true=y_train,
                y_pred=y_fit,
                label="in-sample",
                show_description=False,
            ),
            right=regression_report(
                y_true=y_test,
                y_pred=y_pred,
                label="out-of-sample",
                show_description=True,
            ),
            left_index=True,
            right_index=True,
        )
        # generate difference in percentage between in-sample (train) and out-of-sample (validation) predictions
        .assign(
            **{
                "Difference (%)": lambda x: (x["out-of-sample"] - x["in-sample"])
                / x["out-of-sample"]
                * 100
            }
        )
        # change order of columns
        .loc[:, ["in-sample", "out-of-sample", "Difference (%)", "Description"]]
        .assign(model=model_name)
        # set index name
        .rename_axis(index="metrics")
        .reset_index()
        .set_index(["model", "metrics"])
    )


# function to plot the actual vs predicted values
def plot_actual_vs_predicted_values(
    y_train: pd.Series, y_fit: pd.Series, y_test: pd.Series, y_pred: pd.Series, figsize=(5, 5)
):
    """Plot actual vs predicted values

    Args:
        y_train (pd.Series): the targer variable in the training set
        y_fit (pd.Series): the fitted values in the training set
        y_test (pd.Series): the target variable in the validation set or test set
        y_pred (pd.Series): the predicted values in the validation set or test set
        figsize (tuple, optional): plot size. Defaults to (5,5).
    """

    # Plot pred vs actual values (note we exponentiate the target variable)
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        [y_train.min(), y_train.max()],
        [y_train.min(), y_train.max()],
        "--k",
    )
    ax.scatter(x=y_fit, y=y_train, c='#0B70D4', alpha=0.7)
    ax.scatter(x=y_pred, y=y_test, c='#F9812A', alpha=0.7)

    # add legend
    ax.legend(["Perfect fit", "Train set", "Test set"])
    ax.set_xlabel("Predicted values")
    ax.set_ylabel("True values")
    ax.set_title(
        "Predicted vs Actual values \n r-value train ="
        f" {np.corrcoef(y_train, y_fit)[0,1]:.2f} | r-value test ="
        f" {np.corrcoef(y_test, y_pred)[0,1]:.2f} "
    )

    plt.show()

    return fig, ax