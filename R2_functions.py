# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from typing import Any, Callable, Dict, List, Tuple
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
# from fitter import Fitter, get_common_distributions, get_distributions
import logging

# Check if the data is normally distributed
def normal_check(data: pd.DataFrame) -> pd.DataFrame:
    """Compare the distribution of numeric variables to a normal distribution using the Kolmogrov-Smirnov test,
    where null hypothesis is that the data is normally distributed.
    Wrapper for `scipy.stats.kstest`: the empircal data is compared to a normally distributed variable with the
    same mean and standard deviation. 
    A significant result (p < 0.05) in the goodness of fit test means that the data is not normally distributed.
    
    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe including the columns of interest
    
    Returns
    -------
    df_normality_check: pd.DataFrame
        Dataframe with column names, p-values and an indication of normality
   
    Examples
    --------
    >>> tips = sns.load_dataset("tips")
    >>> df_normality_check = normal_check(tips)
    """
    # Select numeric columns only
    num_features = data.select_dtypes(include='number').columns.tolist()
    # Compare distribution of each feature to a normal distribution with given mean and std
    df_normality_check = data[num_features].apply(
        lambda x: stats.kstest(
            x.dropna(),
            stats.norm.cdf,
            args=(np.nanmean(x), np.nanstd(x)),
            N=len(x),
        )[1],
        axis=0,
    )

    # create a label that indicates whether a feature has a normal distribution or not
    df_normality_check = pd.DataFrame(df_normality_check).reset_index()
    df_normality_check.columns = ["feature", "p-value"]
    df_normality_check["p-value"]=df_normality_check["p-value"].round(3)
    df_normality_check["normality"] = df_normality_check["p-value"] >= 0.05

    return df_normality_check


#Identify outliers based on the inter-quartile range
def iqr_outlier_detector(s: pd.Series, factor: float = 1.5) -> Tuple[List[bool], Any, Any]:
    """Inter-quartile-range-based outlier removal helper.

    Note: if `s` is not numerical, it returns a list with the
    same number of `False` objects as there are rows in `s`.

    Parameters
    ----------
    s: pd.Series
        numerical variable to identify outliers from
    factor: float (default: 1.5)
        determines the upper and lower bounds of what is "normal" in
        the data, anything beyond those bounds is an "anomaly":

        - upper bound: 3rd quartile + (IQR  * factor)

        - lower bound: 1st quartile - (IQR * factor)

    Returns
    -------
    List[bool]
        list of boolean flags indicating if data point is an outlier
    lower: float
        integer of the lower cutoff used to define an outlier
    upper: float
        integer of the upper cutoff used to define an outlier


    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([2, 3, 2.5, 4, 2, 1.5, 3.2, 10])
    >>> is_outlier, lower, upper = iqr_outlier_detector(s, factor=2)
    >>> assert is_outlier == [False, False, False, False, False, False, False, True]

    """
    # only numerical variables
    if s.dtype not in ["float", "int"]:
        return [False] * len(s), np.nan, np.nan

    # calculate inter-quartile range
    q25, q75 = s.quantile(0.25), s.quantile(0.75)
    iqr = q75 - q25

    # calculate the outlier cutoff
    cut_off = iqr * factor
    lower, upper = q25 - cut_off, q75 + cut_off

    # identify outliers
    to_discard = (s < lower) | (s > upper)
    return to_discard.values.tolist(), lower, upper


def std_outlier_detector(s: pd.Series, factor: float = 3.0) -> Tuple[List[bool], Any, Any]:
    """Standard-deviation-based outlier removal helper.

    Note: if `s` is not numerical, it returns a list with the same number of `False`
    objects as there are rows in `s`.

    Parameters
    ----------
    s: pd.Series
        numerical variable to identify outliers from
    factor: float (default: 3.)
        determines the upper and lower bounds of what is "normal" in the data, anything
        beyond those bounds is an "anomaly":

        - upper bound: mean + (std  * factor)

        - lower bound: mean - (std * factor)

    Returns
    -------
    discard list: List[bool]
        list of boolean flags indicating if data point is an outlier
    lower: float
        integer of the lower cutoff used to define an outlier
    upper: float
        integer of the upper cutoff used to define an outlier

    Examples
    --------
    >>> import pandas as pd
    >>> s = pd.Series([2, 3, 2.5, 4, 2, 1.5, 3.2, 10])
    >>> is_outlier, lower, upper = std_outlier_detector(s, factor=1.2)
    >>> assert is_outlier == [False, False, False, False, False, False, False, True]

    """
    # only numerical variables
    if s.dtype not in ["float", "int"]:
        return [False] * len(s), np.nan, np.nan

    # calculate the outlier cutoff
    cut_off = s.std() * factor
    lower, upper = s.mean() - cut_off, s.mean() + cut_off

    # identify outliers
    to_discard = (s < lower) | (s > upper)
    return to_discard.values.tolist(), lower, upper


def univariate_outlier_removal(
    df: pd.DataFrame,
    nonparam_func: Callable = iqr_outlier_detector,
    nonparam_args: Dict[str, Any] = {"factor": 1.5},
    nonparam_name: str = "inter-quartile range, k=1.5",
    std_factor: float = 3.0,
    remove: bool = False,
    plot: bool = False,
    figsize: Tuple[float, float] = (15, 7),
) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
    """Rule-based univariate outlier removal.

    Applies both a parametric or a non-parametric outlier
    removal based on a condition (parametric assumption).
    By default, the parametric method assumes a gaussian
    distribution, and so:

    - if met, defines ouliers based on the empirical rule
    (see https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule)

    - if NOT met, defines ouliers based on its inter-quartile range

    Note that detection is ONLY applied to numerical columns.
    Categorical columns will remain and returned UNALTERED
    in every case.

    Parameters
    ----------
    df: pd.DataFrame
        dataframe with column variables to apply outlier removal to.
        Only numerical variables are considered.

    nonparam_func: callable (default: iqr_outlier_detector())
        non-parametric function. Must take in a Pandas Series,
        and return a list of booleans of the same size
        indicating whether a data point is an outlier or not,
        and the lower and upper float values that define an
        outlier. Default is IQR method.

    nonparam_args: dict (default: {"factor": 1.5})
        non-parametric input arguments. Default is k=1.5 for IQR method.

    nonparam_name: str (default: "non-parametric method")
        non-parametric method name (as shown in summary output).

    std_factor: float (default: 3.)
        factor multiplier for `std_outlier_detector`.
        See 'std_outlier_detector` docstrings for more info.

    remove: bool (default: False)
        whether to apply outlier removal to input dataframe (mask of NaNs).
        Categorical columns remain unaltered.

    plot: bool (default: False)
        whether to plot outliers.

    figsize: Tuple[float, float] (default: (15, 7))
        figure size.

    Returns
    -------
    df: pd.Dataframe
        input dataframe with or without outliers

    summary: pd.DataFrame
        summary report of outlier removal analysis

    fig: plt.Figure (optional)
        if plot==True, return figure object

    Examples
    --------
    >>> import seaborn as sns
    >>> from r2_helpers import univariate_outlier_removal
    >>> df = sns.load_dataset('titanic')
    >>> def some_outlier_detector(s, threshold=4.):
    ...        to_discard = (s < -threshold) | (s > threshold)
    ...        return to_discard.values.tolist(), -threshold, threshold
    >>> df, summary, fig = univariate_outlier_removal(df,
    ...                                               nonparam_func=some_outlier_detector,
    ...                                               nonparam_args={"threshold": 55.},
    ...                                               nonparam_name="hard threshold, t=5",
    ...                                               remove=False,
    ...                                               plot=True,
    ...                                               figsize=(25, 5))

    """
    # make a copy in case remove=True and plot=True
    df = df.copy(deep=True)
    df_original = df.copy(deep=True)

    # check if callable is valid
    discard_list, _, _ = nonparam_func(pd.Series([10.0] * 20), **nonparam_args)

    assert all(
        type(i) == bool for i in discard_list
    ), f"`{nonparam_func.__name__}` callable must return a list of booleans."

    # dictionary to store results
    outliers: Dict[str, list] = {
        "var": [],
        "method": [],
        "outliers-removed": [],
        "outliers-removed %": [],
        "cutoff_lower": [],
        "cutoff_upper": [],
        "index": [],
    }

    fig = None
    if plot:
        # instantiate figure
        fig, ax = plt.subplots(figsize=figsize)

        # store colors per feature method in boxplot
        custom_palette: Dict[str, str] = {}
        param_color = '#4C8BF5'       #blue 
        nonparam_color = '#F9812A'    #orange

    # set default parametric method string
    param_name = f"standard deviation, k={std_factor}"

    # if there are no numerical variables, return original df + empty summaries and plots
    if df.select_dtypes(["int", "float"]).empty:
        return df, pd.DataFrame(outliers), fig

    # loop over every feature
    for i, f in enumerate(df.select_dtypes(["int", "float"])):
        # select feature series
        s = df[f]

        # does it satisfy condition?
        try:
            if normal_check(s.to_frame())["normality"][0]:
                # apply parametric function if condition is met
                to_discard, lower, upper = std_outlier_detector(s, factor=std_factor)

                # log method name
                outliers["method"].append(param_name)

                # log the upper and lower bounds of the cutoff
                outliers["cutoff_lower"].append(lower)
                outliers["cutoff_upper"].append(upper)

                # box color in boxplot
                if plot:
                    custom_palette[f] = param_color
            else:
                # apply non-parametric function if condition is met
                to_discard, lower, upper = nonparam_func(s, **nonparam_args)

                # log method name
                outliers["method"].append(nonparam_name)

                # log the upper and lower bounds of the cutoff
                outliers["cutoff_lower"].append(lower)
                outliers["cutoff_upper"].append(upper)

                # box color in boxplot
                if plot:
                    custom_palette[f] = nonparam_color

        # catch and log exception (if any), and jump the to next
        # column without raising an exception
        except Exception as e:
            logging.exception(f"Exception in `{f}`: {e}")
            continue

        # log summary outliers
        outliers["var"].append(f)
        outliers["outliers-removed"].append(sum(to_discard))
        outliers["outliers-removed %"].append(round(sum(to_discard) / s.notna().sum(), 3) * 100)
        outliers["index"].append(s.reset_index()[to_discard].index.to_list())

        # plot red circles around outliers to remove
        if plot and sum(to_discard) > 0:
            ax.plot(
                s[to_discard].values,
                [i] * sum(to_discard),
                "o",
                ms=13,
                mec="r",
                mfc="none",
                mew=1,
            )

        # remove outliers: mask them with nans
        if remove:
            df[f].iloc[to_discard] = np.nan

    # summary and logs in data frame
    summary = pd.DataFrame(outliers)

    # return (masked) dataframe, summary frame and (if plot) figure handle
    if plot:
        # plot only numerical variables
        ax = sns.boxplot(
            data=df_original.select_dtypes(["int", "float"]),
            fliersize=5,
            orient="h",
            palette=custom_palette,
            ax=ax,
        )

        # add transparency to colors
        for patch, label in zip(ax.artists, summary["method"]):
            r, g, b, a = patch.get_facecolor()
            patch.set_facecolor((r, g, b, 0.4))

        # add legend
        param_patch = Patch(color=param_color, alpha=0.4, label=param_name)
        nonparam_patch = Patch(color=nonparam_color, alpha=0.4, label=nonparam_name)
        ax.legend(handles=[param_patch, nonparam_patch])

    return df, summary, fig
