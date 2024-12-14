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
    explained_variance_score,
    mean_squared_error,
    mean_absolute_error,
    max_error,
)
from IPython.display import display
def plot_residuals(data_dict, dataset_key="validation"):
    """
    Plot residuals distribution (histogram and boxplot) for a given dataset key.

    Parameters:
    - data_dict (dict): Dictionary containing datasets (e.g., "train", "validation").
    - dataset_key (str): Key for the dataset to plot (e.g., "train" or "validation").
    """
    # Check if the dataset_key exists in the dictionary
    if dataset_key not in data_dict:
        raise ValueError(f"Dataset '{dataset_key}' not found in the dictionary.")

    # Extract the dataset
    df = data_dict[dataset_key]

    # Ensure the dataset has the 'residuals' column
    if "residuals" not in df.columns:
        raise ValueError(f"Dataset '{dataset_key}' does not contain 'residuals' column.")

    # Compute summary statistics
    residuals = df["residuals"] / 1000  # Convert residuals to k€
    s_stats = residuals.describe().round(1)

    # Compute mean and std for confidence interval
    mean = s_stats["mean"]
    std_dev = s_stats["std"]
    lower_ci = round(mean - std_dev, 1)  # Mean - 1 Std
    upper_ci = round(mean + std_dev, 1)  # Mean + 1 Std

    # Compute IQR and thresholds for outliers
    IQR = s_stats["75%"] - s_stats["25%"]
    lower_bound = round(s_stats["25%"] - 1.5 * IQR, 1)  # Rounded lower bound
    upper_bound = round(s_stats["75%"] + 1.5 * IQR, 1)  # Rounded upper bound

    # Plot residuals: histogram and boxplot
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

    # 1. Histogram of residuals
    sns.histplot(residuals, kde=True, color="skyblue", ax=axes[0])
    axes[0].axvline(mean, color="red", linestyle="--", label=f"Mean: {mean} k€")
    axes[0].axvline(upper_ci, color="orange", linestyle="--", label=f"Mean + 1 SD: {upper_ci} k€")
    axes[0].axvline(lower_ci, color="orange", linestyle="--", label=f"Mean - 1 SD: {lower_ci} k€")
    axes[0].set_title(f"Residuals Distribution (Histogram) - {dataset_key.capitalize()} Data")
    axes[0].set_xlabel("Residuals (k€)")
    axes[0].set_ylabel("Count")
    axes[0].legend(
        title="Statistics", 
        loc="upper right", 
        labels=[
            f"Mean: {mean} k€",
            f"Std Dev: {std_dev} k€",
            f"Mean ± 1 SD: [{lower_ci}, {upper_ci}] k€"
        ]
    )

    # 2. Boxplot of residuals
    sns.boxplot(x=residuals, ax=axes[1], color="white", boxprops=dict(alpha=0.7), showfliers=False)
    sns.stripplot(x=residuals, ax=axes[1], color="skyblue", alpha=0.6, jitter=True)

    # Highlight areas for outliers
    axes[1].axvspan(residuals.min(), lower_bound, color="orange", alpha=0.2, label=f"Outliers (Low < {lower_bound} k€)")
    axes[1].axvspan(upper_bound, residuals.max(), color="orange", alpha=0.2, label=f"Outliers (High > {upper_bound} k€)")

    # Add lines for the IQR bounds
    axes[1].axvline(lower_bound, color="orange", linestyle="--", label=f"Lower Bound: {lower_bound} k€")
    axes[1].axvline(upper_bound, color="orange", linestyle="--", label=f"Upper Bound: {upper_bound} k€")

    # Configure boxplot aesthetics
    axes[1].set_title(f"Residuals Distribution (Boxplot) - {dataset_key.capitalize()} Data")
    axes[1].set_xlabel("Residuals (k€)")
    axes[1].legend()

    # Tight layout to avoid overlap
    plt.tight_layout()
    plt.show()