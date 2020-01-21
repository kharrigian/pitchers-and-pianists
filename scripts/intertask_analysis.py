
"""
Exploration of the correlation between Skittles and Tapping performance
"""

FIGURE_FMT = ".pdf"
standard_fig = (10,5.8)

##############################
### Imports
##############################

## Standard
import os
import glob
import itertools

## External
import pandas as pd
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
from tqdm import tqdm

# Statistical Modeling/Curve Fitting
import statsmodels.formula.api as smf
from statsmodels.stats import anova
from statsmodels.multivariate import cancorr, manova
from scipy import stats
from scipy.optimize import curve_fit

###############
### Tapping Dependent Variables
###############

def quantile_variation_coefficient(x):
    """
    Compute the Quartile variation coeffient of x

    Args:
        x (array): Array of numeric values

    Returns:
        qvc (float): Quartile variation coefficient
    """
    if x is None or len(x) == 0:
        return np.nan
    q1, q3 = np.nanpercentile(x, [25, 75])
    qvc = (q3 - q1) / (q3 + q1)
    return qvc

###############
### Tapping Dependent Variables
###############

# qvc Variables
qvc_sync_metrics = ["met_qvc_last5"]
qvc_cont_metrics = ["nomet_qvc_last5"]
qvc_metrics = qvc_sync_metrics + qvc_cont_metrics

# Error Variables
error_sync_metrics = ["met_sync_error_last5",
                      "met_sync_error_last5_rel"]
error_cont_metrics = ["nomet_sync_error_last5",
                      "nomet_sync_error_last5_rel"]
error_metrics = error_sync_metrics + error_cont_metrics

# Drift Variables
drift_metrics = ["nomet_drift",
                 "nomet_drift_rel",
                 "nomet_drift_regression"]

# Variable Types
factor_variables = ["trial",
                    "speed_occurrence",
                    "trial_speed",
                    "gender",
                    "musical_experience",
                    "sport_experience"]
continuous_variabless = ["preferred_period",
                         "age"]
all_variables = factor_variables + continuous_variabless

# Fixed Primary Variables
primary_variables = ["age",
                     "gender",
                     "trial",
                     "speed_occurrence",
                     "trial_speed",
                     "musical_experience",
                     "sport_experience"]

##############################
### Load Data
##############################

## Survey Data
survey_data = pd.read_csv("./data/survey.csv")

## Processed Tapping Metrics
processed_tapping_data = pd.read_csv("./data/merged_processed_results.csv")
processed_tapping_data["gender"] = processed_tapping_data["gender"].map(lambda g: "Female" if g == 1 else "Male")
processed_tapping_data["age_bin"] = processed_tapping_data["age_bin"].map(lambda i: ["5-9","10-12","13-19","20-29","30-49","50+"][i])

## Processed Skittles Data
semiprocessed_skittles_data = sio.matlab.loadmat(file_name = "./data/skittles/MuseumSkittlesResults.mat")
subject_index = semiprocessed_skittles_data["skittlesSurvey"][:,0].astype(int)
processed_skittles_data = {}
for skittles_metric, metric_data in semiprocessed_skittles_data.items():
    if skittles_metric.startswith("__") or skittles_metric == "skittlesSurvey":
        continue
    if skittles_metric in ["Error","timingError","timingWindow"]:
        metric_data = metric_data[:, 1:]
    df = pd.DataFrame(data = metric_data,
                      index = subject_index)
    df.index.name = "Subject"
    processed_skittles_data[skittles_metric] = df

## Trials (25 throws, with small breaks in between)
trial_boundaries = np.arange(0,101,25)
blocks = [list(range(x, y)) for x, y in zip(trial_boundaries[:-1], trial_boundaries[1:])]

"""
Skittles Metrics
----------------
Error - Performance Error - Note that 1.0 represents a post hit - Should decrease over tie
timingError - Timing Error - Should decrease over time
timingWindow - Timing Window - Should increase over time
ITI - Inter-throw-Interval - Last column null (since we consider windows) - Want this to converge
qvcITImovingWindow - Moving Window ITI Spread - Want this to converge
"""

##############################
### Metric Summarization
##############################

## Config
remove_post_hits = True

## Process ITIs
itis_df = []
for t, trs in enumerate(blocks):
    trs_agg = processed_skittles_data["ITI"][trs[:-1]].mean(axis=1).reset_index()
    trs_agg["median_ITI"] = processed_skittles_data["ITI"][trs[:-1]].median(axis=1).values
    trs_agg["qvc_ITI"] = processed_skittles_data["ITI"][trs[:-1]].apply(quantile_variation_coefficient, axis = 1)
    trs_agg["sd_ITI"] = processed_skittles_data["ITI"][trs[:-1]].std(axis=1).values
    trs_agg["block"] = t + 1
    trs_agg.rename(columns = {0:"mean_ITI"}, inplace=True)
    itis_df.append(trs_agg)
itis_df = pd.concat(itis_df)

## Process Error
errors_df = []
for t, trs in enumerate(blocks):
    error = processed_skittles_data["Error"][trs]
    post_hits = error.apply(lambda x: [i == 1 or pd.isnull(i) for i in x]).sum(axis=1).values
    if remove_post_hits:
        error = error.apply(lambda x: [i for i in x if i != 1.0 and not pd.isnull(i)], axis = 1)
    error_agg = error.map(np.mean).reset_index()
    error_agg["median_Error"] = error.map(np.median).values
    error_agg["qvc_Error"] = error.map(quantile_variation_coefficient)
    error_agg["sd_Error"] = error.map(np.std).values
    error_agg["post_hits"] = post_hits
    error_agg["block"] = t + 1
    error_agg.rename(columns = {0:"mean_Error"}, inplace = True)
    errors_df.append(error_agg)
errors_df = pd.concat(errors_df)

## Process Timing Error
timing_errors_df = []
for t, trs in enumerate(blocks):
    timing_error =  processed_skittles_data["timingError"][trs]
    timing_error_agg = timing_error.mean(axis = 1).reset_index()
    timing_error_agg["median_timingError"] = timing_error.median(axis = 1).values
    timing_error_agg["qvc_timingError"] = timing_error.apply(quantile_variation_coefficient, axis = 1)
    timing_error_agg["sd_timingError"] = timing_error.std(axis=1).values
    timing_error_agg["block"] = t + 1
    timing_error_agg.rename(columns = {0:"mean_timingError"}, inplace = True)
    timing_errors_df.append(timing_error_agg)
timing_errors_df = pd.concat(timing_errors_df)

## Process Timing Window
timing_window_df = []
for t, trs in enumerate(blocks):
    timing_window =  processed_skittles_data["timingWindow"][trs]
    timing_window_agg = timing_window.mean(axis = 1).reset_index()
    timing_window_agg["median_timingWindow"] = timing_window.median(axis = 1).values
    timing_window_agg["qvc_timingWindow"] = timing_window.apply(quantile_variation_coefficient, axis = 1)
    timing_window_agg["sd_timingWindow"] = timing_window.std(axis=1).values
    timing_window_agg["block"] = t + 1
    timing_window_agg.rename(columns = {0:"mean_timingWindow"}, inplace = True)
    timing_window_df.append(timing_window_agg)
timing_window_df = pd.concat(timing_window_df)
timing_window_df["qvc_timingWindow"] = timing_window_df["sd_timingWindow"] / timing_window_df["mean_timingWindow"]

## Merge Metrics
skittles_data_agg = pd.merge(itis_df, errors_df, on = ["Subject", "block"])
skittles_data_agg = pd.merge(skittles_data_agg, timing_errors_df, on = ["Subject", "block"])
skittles_data_agg = pd.merge(skittles_data_agg, timing_window_df, on = ["Subject", "block"])
skittles_data_agg = skittles_data_agg[["Subject","block"] + [c for c in skittles_data_agg.columns if c not in ["Subject","block"]]]
skittles_data_agg = skittles_data_agg.rename(columns = {"Subject":"subject"})
skittles_data_agg = skittles_data_agg.drop([c for c in skittles_data_agg.columns if c.startswith("sd_")],axis=1)

##############################
### Subject Pool Filtering
##############################

## Subject Overlap
tapping_subjects = set(processed_tapping_data.subject) # 282 subjects
skittles_subjects = set(skittles_data_agg.subject) # 385 subjects
tapping_and_skittles_subjects = tapping_subjects & skittles_subjects # 239 subjects

## Filter Down to Overlap
processed_tapping_data = processed_tapping_data.loc[processed_tapping_data.subject.isin(tapping_and_skittles_subjects)]
processed_tapping_data = processed_tapping_data.reset_index(drop=True).copy()
skittles_data_agg = skittles_data_agg.loc[skittles_data_agg.subject.isin(tapping_and_skittles_subjects)]
skittles_data_agg = skittles_data_agg.reset_index(drop = True).copy()

## Age/Gender/Musical Experience Distribution
age_gender_dist = processed_tapping_data.drop_duplicates("subject").groupby(["age_bin","gender","musical_experience"]).size()

##############################
### Data Formatting and Reshaping
##############################

## Filter Down Skittles to 4th Block
skittles_data_agg = skittles_data_agg.loc[skittles_data_agg.block == 4].reset_index(drop=True).copy()

## Summary Matrices
skittles_X = pd.pivot_table(pd.melt(skittles_data_agg.drop(["block","post_hits"],axis=1),
                                    id_vars=["subject"]),
                            index = "subject",
                            columns = ["variable"])["value"]
tapping_Y = pd.pivot_table(pd.melt(processed_tapping_data[["subject","qvc","drift","error","trial_speed","condition"]],
                                   id_vars = ["subject","trial_speed","condition"]),
                           index = "subject",
                           columns =["variable","trial_speed","condition"])["value"]

## Keep Mean only (Not Median)
X_to_drop = [c for c in skittles_X.columns.tolist() if "median" in c[0]]
skittles_X = skittles_X.drop(X_to_drop, axis=1)

## Unify Column Names
tapping_Y.columns = ["-".join(list(map(str, c))) for c in tapping_Y.columns.tolist()]
# skittles_X.columns = ["-".join(list(map(str,c))) for c in skittles_X.columns.tolist()]

tapping_Y = pd.merge(tapping_Y,
                     processed_tapping_data.set_index("subject")[["preferred_period"]].drop_duplicates(),
                     left_index = True,
                     right_index = True)

## Merge
data_merged = pd.merge(skittles_X,
                       tapping_Y,
                       left_index = True,
                       right_index = True)

##############################
### Correlation Analysis
##############################

## Compute Correlations
corr = data_merged.corr(method = "spearman")
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

## Plot All Correlations
fig, ax = plt.subplots(figsize = (12,8))
sns.heatmap(corr,
            mask=mask,
            cmap=plt.cm.coolwarm,
            vmax=.9,
            center=0,
            square=False,
            linewidths=.5,
            ax = ax,
            xticklabels = corr.index.values,
            yticklabels = corr.index.values,
            cbar_kws={"shrink": .7})
plt.xticks(rotation = 45, ha = "right", fontsize = 7)
plt.yticks(fontsize = 7)
plt.tight_layout()
plt.savefig("./plots/intertask/full_correlation_matrix" + FIGURE_FMT)
plt.savefig("./plots/intertask/full_correlation_matrix" + ".png")
plt.close()

## Plot Cross Correlations
fig, ax = plt.subplots(figsize = (12, 8))
sns.heatmap(corr.loc[tapping_Y.columns, skittles_X.columns],
            yticklabels=tapping_Y.columns.tolist(),
            xticklabels=skittles_X.columns.tolist(),
            cmap=plt.cm.coolwarm,
            linewidths=.5,
            ax = ax,
            annot = corr.loc[tapping_Y.columns, skittles_X.columns].values,
            fmt = ".2f",
            annot_kws={"size": 8})
plt.xticks(rotation = 45, ha = "right", fontsize = 7)
plt.yticks(fontsize = 7)
plt.tight_layout()
plt.savefig("./plots/intertask/intertask_correlation_matrix" + FIGURE_FMT)
plt.savefig("./plots/intertask/intertask_correlation_matrix" + ".png")
plt.close()

## Prepare to Plot
corr_dir = "./plots/intertask/correlation_scatters/"
if not os.path.exists(corr_dir):
    os.makedirs(corr_dir)
else:
    res = os.system("rm -rf {}".format(corr_dir))
    os.makedirs(corr_dir)

## Plot Top Correlations
n_top = 20
tapping_mets = tapping_Y.drop(["preferred_period"],axis=1).columns.tolist()
skittles_mets = ["mean_timingError","mean_timingWindow", "qvc_ITI"]
top_correlations = corr.loc[tapping_mets, skittles_mets].unstack().map(np.abs).reset_index().sort_values(0,ascending = False)[["level_0","level_1"]].values[:n_top]
for skit, tap in top_correlations:
    data_to_plot = data_merged[[skit, tap]].dropna()
    fig, ax = plt.subplots(1, 2, figsize = (12, 8))
    data_to_plot.plot.scatter(skit, tap, ax = ax[0])
    data_to_plot.rank(axis=0).plot.scatter(skit, tap, ax = ax[1])
    pearson_corr = pearsonr(data_to_plot[skit], data_to_plot[tap])
    spearman_corr = spearmanr(data_to_plot[skit], data_to_plot[tap])
    ax[0].set_title("Pearson Correlation: {:,.2f} (p={:,.2f})".format(pearson_corr[0], pearson_corr[1]))
    ax[1].set_title("Spearman Rank Correlation: {:,.2f} (p={:,.2f})".format(spearman_corr[0], spearman_corr[1]))
    ax[1].set_xlabel("{} subject rank".format(skit))
    ax[1].set_ylabel("{} subject rank".format(tap))
    fig.tight_layout()
    fig.savefig(corr_dir + "{}--{}".format(skit, tap) + FIGURE_FMT)
    fig.savefig(corr_dir + "{}--{}".format(skit, tap) + ".png")
    plt.close()

##############################
### Figure Visuals
##############################

## Merge in Subject Metadata
data_merged_ols = pd.merge(data_merged,
                           processed_tapping_data.drop_duplicates("subject").set_index("subject")[
                               ["age",
                                "age_bin",
                                "gender",
                                "healthy",
                                "musical_experience",
                                "musical_experience_yrs",
                                "sport_experience",
                                "sport_experience_yrs"]
                           ],
                           left_index = True,
                           right_index = True)
data_merged_ols.rename(columns = dict((col, col.replace("-","_")) for col in data_merged_ols.columns),
                       inplace = True)

## Plotting Function
def plot_comparison(tapping_met = "qvc",
                    tap_met_name = "$QVC_{ITI}$",
                    segment = "unpaced",
                    plot_rank = True):
    """
    Args:
        tapping_met (str): Which tap metric to plot (qvc, error, drift)
        tap_met_name (str): Name of the tap metric to be included in the plot title
        segment (str): "unpaced" or "paced"
        plot_rank (bool): If True, plot the rank order of the metrics instead of raw values
    
    Returns:
        fig, ax (matplotlib objects
    """
    skittles_mets = ["mean_timingError","mean_timingWindow","qvc_ITI"]
    fig, ax = plt.subplots(3, 3, figsize = standard_fig, sharex = plot_rank, sharey = True)
    for t, (trial_speed, tlbl) in enumerate(zip(["SpedUp","NoChange","SlowedDown"],["80%","100%","120%"])):
        for m, (skit_met, sklbl) in enumerate(zip(skittles_mets,["Timing Error", "Timing Window", "$QVC_{IRI}$"])):
            plot_ax = ax[t,m]
            tmet = "{}_{}_{}".format(tapping_met, trial_speed, segment)
            if plot_rank:
                x = data_merged_ols[skit_met].rank(ascending = True, method = "dense")
                y = data_merged_ols[tmet].rank(ascending = True, method = "dense")
            else:
                x = data_merged_ols[skit_met]
                y = data_merged_ols[tmet]
            plot_ax.scatter(x,
                            y,
                            s = 25,
                            alpha = .3,
                            color = "slategray")
            if plot_rank:
                plot_ax.set_xlim(min(x) - 3, max(x) + 3)
                plot_ax.set_ylim(min(y) - 3, max(y) + 3)
            if t == 2:
                plot_ax.set_xlabel(sklbl, fontsize = 12)
            else:
                if not plot_rank:
                    plot_ax.set_xticks([])
        ax[t][0].set_ylabel(tlbl + "\nPreferred Period", fontsize = 12, labelpad = 10)
    fig.text(0.55,
            0.02,
            "Throwing Performance" + {True:" (Rank)",False:""}[plot_rank],
            fontweight = "bold",
            horizontalalignment="center",
            fontsize = 16)
    fig.text(0.04,
            0.55,
            "Tapping Performance{}".format({True:" (Rank)",False:""}[plot_rank]),
            fontweight = "bold",
            horizontalalignment = "center",
            verticalalignment = "center",
            rotation = 90,
            fontsize = 16)
    fig.suptitle("Tapping Metric: {}".format(tap_met_name),
                fontsize = 16,
                x = 0.15,
                y = .95,
                fontweight = "bold",
                horizontalalignment = "left",
                verticalalignment = "center")
    fig.tight_layout()
    fig.subplots_adjust(bottom = 0.15, left = 0.15, hspace = 0.10, top = .92)
    return fig, ax

## Run Plotting
tap_metrics = [("drift","Drift"),("qvc","$QVC_{ITI}$"),("error","Timing Error")]
for tap_met, met_name in tap_metrics:
    fig, ax = plot_comparison(tap_met, met_name, plot_rank = True)
    fig.savefig("./plots/intertask/{}_correlations".format(tap_met) + FIGURE_FMT)
    fig.savefig("./plots/intertask/{}_correlations".format(tap_met) + ".png")
    plt.close()

##############################
### Miscellaneous Visuals
##############################

## Preferred Period vs Inter-Release-Interval
fig, ax = plt.subplots(figsize = standard_fig)
data_merged_ols.plot.scatter("preferred_period",
                             "mean_ITI",
                             ax = ax,
                             color = "slategray",
                             s = 40,
                             alpha = .3,
                             edgecolor = "black")
ax.set_xlabel("Preferred Period (ms)",
              fontsize = 16,
              fontweight = "bold",
              labelpad = 10)
ax.set_ylabel("Mean Inter-Release-Interval (ms)",
              fontsize = 16,
              fontweight = "bold",
              labelpad = 10)
ax.tick_params(labelsize = 14)
fig.tight_layout()
fig.savefig("./plots/intertask/preferred_period_IRI_scatter" + FIGURE_FMT)
fig.savefig("./plots/intertask/preferred_period_IRI_scatter" + ".png")
plt.close()

## Preferred Period/Mean ITI
pp_iri_corr =  spearmanr(data_merged_ols["preferred_period"], data_merged_ols["mean_ITI"])

##############################
### OLS ANOVA for Rythmicity
##############################

## Fit OLS Model
formula = "qvc_ITI ~ age + C(gender) + C(musical_experience) + C(sport_experience)"
ols_model = smf.ols(formula, data_merged_ols).fit()
aov_results = anova.anova_lm(ols_model)