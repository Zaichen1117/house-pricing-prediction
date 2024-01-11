# -*- coding: utf-8 -*-
from itertools import combinations, product
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import alexandergovern
from R2_functions import normal_check

def chain_snap(data, fn=lambda x: x.shape, msg=None):
    """Print things in method chaining, leaving the dataframe untouched.
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




def one_way_ANOVA(
    data: pd.DataFrame,
    feature: str,
    grouping_var: str,
    groups_of_interest: list,
    show=False,
    plot=False,
    figsize=(11.7, 8.27),
    col_wrap=None,
):
    """Run one-way ANOVAs using `scipy.stats.f_oneway` and check homogeneity of variances with Levenes test using `scipy.stats.levene`.

    `one_way_ANOVA` assumes equal variances within the groups and will not give a warning if show=False.

    Parameters
    ----------
    data: pandas.DataFrame)
        Dataframe with `feature` and `grouping_var` in columns
    feature: str
        Name of the feature
    grouping_var: str
        Name of the  column with grouping labels in `data`
    groups_of_interest: list
        Names (str) of labels in `data[grouping_var]`
    show: bool
        whether to print the results
    plot: bool
        whether to plot the distribution and the data
    figsize: tuple (default: (11.7, 8.27))
        Width and height of the figure in inches
    col_wrap: int or None (default: None)
        If int, number of subplots that are allowed in a single row

    Returns
    -------
    df_result: pd.DataFrame
    df_descriptive: pd.DataFrame
    distplot: Figure
        Figure if plot == True, else None
    boxplot: Figure
        Figure if plot == True, else None

    Examples
    --------
    >>> import seaborn as sns
    >>> tips = sns.load_dataset("tips")
    >>> _, _, _, _ = one_way_ANOVA(tips, 'tip', 'day', ['Sat','Sun','Thur'], show = True, plot = False)

    """
    # select the 'feature' and 'grouping_var' columns and remove row if any nan present
    data = data.copy()
    data = data[[feature, grouping_var]].dropna(axis=0, how="any")

    # Raise error if feature is not numeric
    if feature not in data.select_dtypes("number").columns:
        raise TypeError(f"Feature {feature} should be numeric")

    # select the groups of interest and remove any not used category from the categorical index
    data = data.loc[data[grouping_var].isin(groups_of_interest), :]
    if data[grouping_var].dtype.name == "category":
        data[grouping_var] = data[grouping_var].cat.remove_unused_categories()

    data[grouping_var] = data[grouping_var].astype('category').cat.as_ordered(inplace=False)
    data = data.groupby(grouping_var).filter(lambda x: len(x) > 1)


    # get descriptive values, keep only interested rows
    df_descriptive = data.groupby(grouping_var, observed=True)[feature].describe()
    _ = df_descriptive.reset_index(inplace=True)

    # Raise warning if groups of interest not in the dataframe
    if not all(grp in df_descriptive[grouping_var].values.tolist() for grp in groups_of_interest):
        warnings.warn(
            f"One of the groups did not have any observations for {feature}",
            stacklevel=2,
        )

    values_per_group = {
        grp_label: values
        for grp_label, values in data.groupby(grouping_var, observed=True)[feature]
    }

    # Check assumption: homogeneity of variances
    (levene, levene_p_value) = stats.levene(*values_per_group.values())

    if levene_p_value > 0.05:
        # Equal variances:
        variance_outcome = "Equal"
        trust_results = "ANOVA"
        # Run one way ANOVA
        (f_value, p_value) = stats.f_oneway(*values_per_group.values())
    else:
        # Unequal variances: ANOVA cannot be trusted
        variance_outcome = "Unequal"
        trust_results = "Alexander-Govern"
        # Run Alexander-Govern
        ag = alexandergovern(*values_per_group.values())
        f_value = ag.statistic
        p_value = ag.pvalue
    

    # Lakens, D.(2013).Calculating and reporting effect sizes to facilitate cumulative science:
    # a practical primer for t - tests and ANOVAs.Frontiers in psychology, 4, 863.
    # eta_squared = ((f * df_effect) / ((f * df_effect) + df_error))

    df_effect = len(groups_of_interest) - 1
    df_error = data[feature].count() - df_effect
    eta_squared = (f_value * df_effect) / ((f_value * df_effect) + df_error)

    if show:
        print(
            f"=== One-way anova: variable = *{feature}* | groups ="
            f" *{', '.join(groups_of_interest)}* defined in *{grouping_var}*"
            " ===\n"
        )
        print("Missing values are dropped\n")

        # Describe the samples
        print(df_descriptive)
        print("\n")

        # Print results Levenes test
        print("Levenes test for homogeneity of variances (H0 = homogeneity):")
        print(f"- W = {levene:.2f}")
        print(f"- p-value = {levene_p_value:.3f}")

        if levene_p_value > 0.05:
            # Equal variances:
            print("- Equal variances detected \n")
        else:
            print(
                "- Unequal variances detected by Levenes test, so ANOVA results"
                " might be untrustworthy"
            )

        # Print results ANOVA
        print("Outcome ANOVA: ")
        print(f"- F-value = {f_value:.2f}")
        print(f"- df_effect = {df_effect}")
        print(f"- df_error = {df_error}")
        print(f"- p-value = {p_value:.3f}")

        if p_value < 0.05:
            print("- Statistical significance detected")
        else:
            print("- Statistical significance NOT detected")
        print("\n")

    distplot = None
    boxplot = None

    if plot:
        # Plot the data
        boxplot, ax = plt.subplots(figsize=figsize)
        _ = sns.boxplot(ax=ax, x=grouping_var, y=feature, data=data)
        _ = sns.swarmplot(
            ax=ax,
            x=grouping_var,
            y=feature,
            data=data,
            alpha=0.50,
            size=2,
        )
        _ = ax.set_title(f"Boxplot {feature} across {grouping_var}")
        plt.xticks(rotation=90)

    dict_result = {
        "test-type": "one way ANOVA",
        "feature": feature,
        "group-var": grouping_var,
        "f-value": round(f_value, 3),
        "eta-squared": round(eta_squared, 3),
        "df-effect": int(df_effect),
        "df-error": int(df_error),
        "p-value": round(p_value, 3),
        "stat-sign": (p_value < 0.05),
        "variance": variance_outcome,
        "results": trust_results,
    }

    df_result = pd.DataFrame(data=dict_result, index=[0])

    return df_result, df_descriptive, distplot, boxplot



# def one_way_ANOVA(
#     data: pd.DataFrame,
#     feature: str,
#     grouping_var: str,
#     groups_of_interest: list,
#     show=False,
#     plot=False,
#     figsize=(11.7, 8.27),
#     col_wrap=None,
# ):
#     """Run one-way ANOVAs using `scipy.stats.f_oneway` and check homogeneity of variances with Levenes test using `scipy.stats.levene`.

#     `one_way_ANOVA` assumes equal variances within the groups and will not give a warning if show=False.

#     Parameters
#     ----------
#     data: pandas.DataFrame)
#         Dataframe with `feature` and `grouping_var` in columns
#     feature: str
#         Name of the feature
#     grouping_var: str
#         Name of the  column with grouping labels in `data`
#     groups_of_interest: list
#         Names (str) of labels in `data[grouping_var]`
#     show: bool
#         whether to print the results
#     plot: bool
#         whether to plot the distribution and the data
#     figsize: tuple (default: (11.7, 8.27))
#         Width and height of the figure in inches
#     col_wrap: int or None (default: None)
#         If int, number of subplots that are allowed in a single row

#     Returns
#     -------
#     df_result: pd.DataFrame
#     df_descriptive: pd.DataFrame
#     distplot: Figure
#         Figure if plot == True, else None
#     boxplot: Figure
#         Figure if plot == True, else None

#     Examples
#     --------
#     >>> import seaborn as sns
#     >>> tips = sns.load_dataset("tips")
#     >>> _, _, _, _ = one_way_ANOVA(tips, 'tip', 'day', ['Sat','Sun','Thur'], show = True, plot = False)

#     """
#     # select the 'feature' and 'grouping_var' columns and remove row if any nan present
#     data = data.copy()
#     data = data[[feature, grouping_var]].dropna(axis=0, how="any")

#     # Raise error if feature is not numeric
#     if feature not in data.select_dtypes("number").columns:
#         raise TypeError(f"Feature {feature} should be numeric")

#     # select the groups of interest and remove any not used category from the categorical index
#     data = data.loc[data[grouping_var].isin(groups_of_interest), :]
#     if data[grouping_var].dtype.name == "category":
#         data[grouping_var] = data[grouping_var].cat.remove_unused_categories()

#     # get descriptive values, keep only interested rows
#     df_descriptive = data.groupby(grouping_var, observed=True)[feature].describe()
#     _ = df_descriptive.reset_index(inplace=True)

#     # Raise warning if groups of interest not in the dataframe
#     if not all(grp in df_descriptive[grouping_var].values.tolist() for grp in groups_of_interest):
#         warnings.warn(
#             f"One of the groups did not have any observations for {feature}",
#             stacklevel=2,
#         )

#     values_per_group = {
#         grp_label: values
#         for grp_label, values in data.groupby(grouping_var, observed=True)[feature]
#     }

#     # Check assumption: homogeneity of variances
#     (levene, levene_p_value) = stats.levene(*values_per_group.values())

#     if levene_p_value > 0.05:
#         # Equal variances:
#         variance_outcome = "Equal"
#         trust_results = "trustworthy"
#     else:
#         # Unequal variances: ANOVA cannot be trusted
#         variance_outcome = "Unequal"
#         trust_results = "untrustworthy"

#     # Run one way ANOVA
#     (f_value, p_value) = stats.f_oneway(*values_per_group.values())

#     # Lakens, D.(2013).Calculating and reporting effect sizes to facilitate cumulative science:
#     # a practical primer for t - tests and ANOVAs.Frontiers in psychology, 4, 863.
#     # eta_squared = ((f * df_effect) / ((f * df_effect) + df_error))

#     df_effect = len(groups_of_interest) - 1
#     df_error = data[feature].count() - df_effect
#     eta_squared = (f_value * df_effect) / ((f_value * df_effect) + df_error)

#     if show:
#         print(
#             f"=== One-way anova: variable = *{feature}* | groups ="
#             f" *{', '.join(groups_of_interest)}* defined in *{grouping_var}*"
#             " ===\n"
#         )
#         print("Missing values are dropped\n")

#         # Describe the samples
#         print(df_descriptive)
#         print("\n")

#         # Print results Levenes test
#         print("Levenes test for homogeneity of variances (H0 = homogeneity):")
#         print(f"- W = {levene:.2f}")
#         print(f"- p-value = {levene_p_value:.3f}")

#         if levene_p_value > 0.05:
#             # Equal variances:
#             print("- Equal variances detected \n")
#         else:
#             print(
#                 "- Unequal variances detected by Levenes test, so ANOVA results"
#                 " might be untrustworthy"
#             )

#         # Print results ANOVA
#         print("Outcome ANOVA: ")
#         print(f"- F-value = {f_value:.2f}")
#         print(f"- df_effect = {df_effect}")
#         print(f"- df_error = {df_error}")
#         print(f"- p-value = {p_value:.3f}")

#         if p_value < 0.05:
#             print("- Statistical significance detected")
#         else:
#             print("- Statistical significance NOT detected")
#         print("\n")

#     distplot = None
#     boxplot = None

#     if plot:
#         # Plot the data
#         boxplot, ax = plt.subplots(figsize=figsize)
#         _ = sns.boxplot(ax=ax, x=grouping_var, y=feature, data=data)
#         _ = sns.swarmplot(
#             ax=ax,
#             x=grouping_var,
#             y=feature,
#             data=data,
#             alpha=0.50,
#             size=2,
#         )
#         _ = ax.set_title(f"Boxplot {feature} across {grouping_var}")
#         plt.xticks(rotation=90)

#     dict_result = {
#         "test-type": "one way ANOVA",
#         "feature": feature,
#         "group-var": grouping_var,
#         "f-value": round(f_value, 3),
#         "eta-squared": round(eta_squared, 3),
#         "df-effect": int(df_effect),
#         "df-error": int(df_error),
#         "p-value": round(p_value, 3),
#         "stat-sign": (p_value < 0.05),
#         "variance": variance_outcome,
#         "results": trust_results,
#     }

#     df_result = pd.DataFrame(data=dict_result, index=[0])

#     return df_result, df_descriptive, distplot, boxplot




def correlation_analysis(
    data: pd.DataFrame,
    col_list=None,
    row_list=None,
    check_norm=False,
    method: str = "pearson",
    dropna: str = "pairwise",
) -> dict:
    r"""Run correlations for numerical features and return output in different formats.
    Different methods to compute correlations and to handle missing values are implemented.
    Inspired by `researchpy.corr_case` and `researchpy.corr_pair`.
    Parameters
    ----------
    data : pandas.DataFrame
        Dataframe with variables in columns, cases in rows
    row_list: list or None (default: None)
        List with names of columns in `data` that should be in the rows of the correlogram.
        If None, all columns are used but only every unique combination.
    col_list: list or None (default: None)
        List with names of columns in `data` that should be in the columns of the correlogram.
        If None, all columns are used and only every unique combination.
    check_norm: bool (default: False)
        If True, normality will be checked for columns in `data` using `normal_check`.
        This influences the used method for correlations, i.e. Pearson
        or Spearman. Note: normality check ignores missing values.
    method: {'pearson', 'kendall', 'spearman'}, default 'pearson'
        Type of correlation, either Pearson's r, Spearman's rho, or Kendall's tau,
        implemented via respectively
        `scipy.stats.pearsonr`, `scipy.stats.spearmanr`, and `scipy.stats.kendalltau`
        Will be ignored if check_norm=True. Instead, Person's r is used
        for every combination of normally distributed
        columns and Spearman's rho is used for all other combinations.
    dropna : {'listwise', 'pairwise'}, default 'pairwise'
        Should rows with missing values be dropped over the complete
        `data` ('listwise') or for every correlation
        separately ('pairwise')


    Returns
    -------
    result_dict: dict
    Dictionary containing with the following keys:
    info: pandas.DataFrame
        Description of correlation method, missing values handling
        and number of observations
    r-values: pandas.DataFrame
        Dataframe with correlation coefficients. Indices and columns
        are column names from `data`. Only lower
        triangle is filled.
    p-values: pandas.DataFrame
        Dataframe with p-values. Indices and columns are column names
        from `data`. Only lower triangle is filled.
    N: pandas.DataFrame
        Dataframe with numbers of observations. Indices and columns
        are column names from `data`. Only lower
        triangle is filled. If dropna ='listwise', every correlation
        will have the same number of observations.
    summary: pandas.DataFrame
        Dataframe with columns ['analysis', 'feature1', 'feature2',
        'r-value', 'p-value', 'N', 'stat-sign']
        which indicate the type of test used for the correlation,
        the pair of columns, the correlation coefficient,
        the p-value, the number of observations for each combination
        of columns in `data` and whether the r-value is
        statistically significant.

    Examples
    --------
    >>> import seaborn as sns
    >>> iris = sns.load_dataset('iris')
    >>> dict_results = correlation_analysis(iris,
    ...                                     method='pearson',
    ...                                     dropna='listwise',
    ...                                     check_norm=True)
    >>> dict_results['summary']
    References
    ----------
    Bryant, C (2018). researchpy's documentation [Revision 9ae5ed63]. Retrieved from
    https://researchpy.readthedocs.io/en/latest/
    """

    # Settings test
    if method == "pearson":
        test, test_name = stats.pearsonr, "Pearson"
    elif method == "spearman":
        test, test_name = stats.spearmanr, "Spearman Rank"
    elif method == "kendall":
        test, test_name = stats.kendalltau, "Kendall's Tau-b"
    else:
        raise ValueError("method not in {'pearson', 'kendall', 'spearman'}")

    # Copy numerical data from the original data
    data = data.copy().select_dtypes("number")

    # Get correct lists
    if col_list and not row_list:
        row_list = data.select_dtypes("number").drop(col_list, axis=1).columns.tolist()
    elif row_list and not col_list:
        col_list = data.select_dtypes("number").drop(row_list, axis=1).columns.tolist()

    # Initializing dataframes to store results
    info = pd.DataFrame()
    summary = pd.DataFrame()
    if not col_list and not row_list:
        r_vals = pd.DataFrame(columns=data.columns, index=data.columns)
        p_vals = pd.DataFrame(columns=data.columns, index=data.columns)
        n_vals = pd.DataFrame(columns=data.columns, index=data.columns)
        iterator = combinations(data.columns, 2)  # type: ignore
    else:
        r_vals = pd.DataFrame(columns=col_list, index=row_list)
        p_vals = pd.DataFrame(columns=col_list, index=row_list)
        n_vals = pd.DataFrame(columns=col_list, index=row_list)
        iterator = product(col_list, row_list)  # type: ignore

    if dropna == "listwise":
        # Remove rows with missing values
        data = data.dropna(how="any", axis="index")
        info = pd.concat(
            [
                info,
                pd.DataFrame(
                    {
                        f"{test_name} correlation test using {dropna} deletion": (
                            f"Total observations used = {len(data)}"
                        )
                    },
                    index=[0],
                ),
            ]
        )
    elif dropna == "pairwise":
        info = pd.concat(
            [
                info,
                pd.DataFrame(
                    {
                        f"{test_name} correlation test using {dropna} deletion": (
                            f"Observations in the data = {len(data)}"
                        )
                    },
                    index=[0],
                ),
            ]
        )
    else:
        raise ValueError("dropna not in {'listwise', 'pairwise'}")

    if check_norm:
        # Check normality of all columns in the data
        df_normality = normal_check(data)
        norm_names = df_normality.loc[df_normality["normality"], "feature"].tolist()

    # Iterating through the Pandas series and performing the correlation
    for col1, col2 in iterator:
        if dropna == "pairwise":
            # Remove rows with missing values in the pair of columns
            test_data = data[[col1, col2]].dropna()
        else:
            test_data = data

        if check_norm:
            # Select Pearson's r only if both columns are normally distributed
            if (col1 in norm_names) and (col2 in norm_names):
                test, test_name = stats.pearsonr, "Pearson"
            else:
                test, test_name = stats.spearmanr, "Spearman Rank"

        # Run correlations
        r_value, p_value = test(test_data.loc[:, col1], test_data.loc[:, col2])
        n_value = len(test_data)

        # Store output in matrix format
        try:
            r_vals.loc[col2, col1] = r_value
            p_vals.loc[col2, col1] = p_value
            n_vals.loc[col2, col1] = n_value
        except KeyError:
            r_vals.loc[col1, col2] = r_value
            p_vals.loc[col1, col2] = p_value
            n_vals.loc[col1, col2] = n_value

        # Store output in dataframe format
        dict_summary = {
            "analysis": test_name,
            "feature1": col1,
            "feature2": col2,
            "r-value": r_value,
            "p-value": p_value,
            "stat-sign": (p_value < 0.05),
            "N": n_value,
        }

        summary = pd.concat(
            [summary, pd.DataFrame(data=dict_summary, index=[0])],
            axis=0,
            ignore_index=True,
            sort=False,
        )

    # Embed results within a dictionary
    result_dict = {
        "r-value": r_vals,
        "p-value": p_vals,
        "N": n_vals,
        "info": info,
        "summary": summary,
    }

    return result_dict




def plot_correlogram(
    data: pd.DataFrame,
    row_list=None,
    col_list=None,
    check_norm=False,
    method="pearson",
    dropna="pairwise",
    margins=None,
    font_scale=1.2,
    show_p=True,
    cmap=sns.diverging_palette(h_neg=10, h_pos=240, as_cmap=True),
    figsize=(15, 15),
):
    """Plot correlogram of numerical features.

    Rows with missing values are excluded. Different methods to
    compute correlations are implemented.

    Parameters
    ----------
    data: pandas.DataFrame
        Dataframe with variables in columns, cases in rows
    row_list: list or None (default: None)
        List with names of columns in `data` that should be
        in the rows of the correlogram.
        If None, all columns are used and only the lower
        half of the correlogram will be filled.
    col_list: list or None (default: None)
        List with names of columns in `data` that should be
        in the columns of the correlogram.
        If None, all columns are used and only the lower
        half of the correlogram will be filled.
    check_norm: bool (default: False)
        If True, normality will be checked for columns
        in `data` using `normal_check`. This influences the used method
        for correlations, `method` will be ignored.
        Note: normality check ignores missing values.
    method: {'pearson', 'kendall', 'spearman'}, default 'pearson'
        Type of correlation, either Pearson's r, Spearman's rho,
        or Kendall's tau, implemented via respectively
        `scipy.stats.pearsonr`, `scipy.stats.spearmanr`,
        and `scipy.stats.kendalltau`. Ignored if check_norm is True.
    dropna : {'listwise', 'pairwise'}, default 'pairwise'
        Should rows with missing values be dropped over
        the complete `data` ('listwise') or for every correlation
        separately ('pairwise')
    margins: dict or 'jupyter' or None (default: None)
        Margins for the correlogram. Any of them that are None
        are referred from `matplotlib.pyplot.subplots_adjust`
        If 'jupyter', default values are {'left': None, 'bottom': 1,
        'right': None, 'top': 2}.
    font_scale: float
        Size of the labels in the correlogram.
    show_p: bool (default: True)
        Place crosses when correlation is not significant
        (i.e. p-value higher than 0.05).
    cmap: colormap (either seaborn or matplotlib)
        A continuous colormap either from seaborn or matplotlib
        which will be used to define the extremes in the
        correlogram. For more information see:
        https://seaborn.pydata.org/tutorial/color_palettes.html
    figsize: tuple (default: (15, 15))
        Width and height of the figure in inches.

    Returns
    -------
    corplot: Figure
        Graph with `seaborn.heatmap` of the correlations (lower triangle only)

    Examples
    --------
    >>> iris = sns.load_dataset('iris')
    >>> _ = plot_correlogram(iris, method='pearson')

    """
    # Compute correlation matrix
    dict_results = correlation_analysis(
        data,
        col_list=col_list,
        row_list=row_list,
        check_norm=check_norm,
        method=method,
        dropna=dropna,
    )
    corr = dict_results["r-value"]
    corr = corr.astype("float64")

    if col_list or row_list:
        mask = None
    else:
        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

    # Add the mask to the heatmap
    corplot, ax = plt.subplots(figsize=figsize)
    _ = sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        cmap=cmap,
        center=0,
        linewidths=1,
        annot=True,
        fmt=".3f",
        vmin=-1,
        vmax=1,
    )

    if show_p:
        pvalues = dict_results["p-value"].values
        # Set X where pvalues is bigger than 0.05
        pvalues_str = np.where(pvalues < 0.05, "", "X")

        if col_list or row_list:
            # Run over all elements of the array
            iterator = np.ndindex(pvalues.shape)
        else:
            # Only take one half of the pvalues
            iterator = combinations(range(pvalues.shape[0]), 2)

        for y, x in iterator:
            if not col_list and not row_list:
                # Reverse x and y to make sure the crosses are plotted at the right places
                x, y = y, x
            _ = plt.text(
                x + 0.5,
                y + 0.5,
                pvalues_str[y, x],
                horizontalalignment="center",
                verticalalignment="center",
                color="gray",
                fontsize=font_scale * 30,
            )

    if check_norm:
        _ = ax.set_title("Correlation using Pearson and Spearman")
    else:
        _ = ax.set_title(f"{method.capitalize()} correlation")

    # Move axis to make sure they align
    ymax, ymin = plt.gca().get_ylim()
    _ = plt.gca().set_ylim(bottom=ymin, top=ymax)

    if not margins:
        margins = {"left": None, "bottom": None, "right": None, "top": None}
    elif margins == "jupyter":
        margins = {"left": None, "bottom": 1, "right": None, "top": 2}

    _ = plt.subplots_adjust(**margins)

    return corplot
