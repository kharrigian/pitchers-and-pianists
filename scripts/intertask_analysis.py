
"""
Exploration of the correlation between Skittles and Tapping performance
"""

FIGURE_FMT = ".pdf"

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

# Statistical Modeling/Curve Fitting
import statsmodels.formula.api as smf
from statsmodels.multivariate import cancorr, manova
from scipy import stats
from scipy.optimize import curve_fit

###############
### Tapping Dependent Variables
###############

# qvc Variables
qvc_sync_metrics = ["met_qvc_last5"]
qvc_cont_metrics = ["nomet_qvc_last5"]
qvc_metrics = qvc_sync_metrics + qvc_cont_metrics

# Error Variables
error_sync_metrics = ["met_sync_error_last5", "met_sync_error_last5_rel"]
error_cont_metrics = ["nomet_sync_error_last5","nomet_sync_error_last5_rel"]
error_metrics = error_sync_metrics + error_cont_metrics

# Drift Variables
drift_metrics = ["nomet_drift","nomet_drift_rel", "nomet_drift_regression"]

# Variable Types
factor_variables = ["trial","speed_occurrence","trial_speed", "gender", "musical_experience", "sport_experience"]
continuous_variabless = ["preferred_period","age"]
all_variables = factor_variables + continuous_variabless

# Fixed Primary Variables
primary_variables = ["age","gender","trial","speed_occurrence","trial_speed","musical_experience","sport_experience"]

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
    trs_agg["sd_ITI"] = processed_skittles_data["ITI"][trs[:-1]].std(axis=1).values
    trs_agg["block"] = t + 1
    trs_agg.rename(columns = {0:"mean_ITI"}, inplace=True)
    itis_df.append(trs_agg)
itis_df = pd.concat(itis_df)
itis_df["qvc_ITI"] = itis_df["sd_ITI"] / itis_df["mean_ITI"]

## Process Error
errors_df = []
for t, trs in enumerate(blocks):
    error = processed_skittles_data["Error"][trs]
    post_hits = error.apply(lambda x: [i == 1 or pd.isnull(i) for i in x]).sum(axis=1).values
    if remove_post_hits:
        error = error.apply(lambda x: [i for i in x if i != 1.0 and not pd.isnull(i)], axis = 1)
    error_agg = error.map(np.mean).reset_index()
    error_agg["median_Error"] = error.map(np.median).values
    error_agg["sd_Error"] = error.map(np.std).values
    error_agg["post_hits"] = post_hits
    error_agg["block"] = t + 1
    error_agg.rename(columns = {0:"mean_Error"}, inplace = True)
    errors_df.append(error_agg)
errors_df = pd.concat(errors_df)
errors_df["qvc_Error"] = errors_df["sd_Error"] / errors_df["mean_Error"]

## Process Timing Error
timing_errors_df = []
for t, trs in enumerate(blocks):
    timing_error =  processed_skittles_data["timingError"][trs]
    timing_error_agg = timing_error.mean(axis = 1).reset_index()
    timing_error_agg["median_timingError"] = timing_error.median(axis = 1).values
    timing_error_agg["sd_timingError"] = timing_error.std(axis=1).values
    timing_error_agg["block"] = t + 1
    timing_error_agg.rename(columns = {0:"mean_timingError"}, inplace = True)
    timing_errors_df.append(timing_error_agg)
timing_errors_df = pd.concat(timing_errors_df)
timing_errors_df["qvc_timingError"] = timing_errors_df["sd_timingError"] / timing_errors_df["mean_timingError"]

## Process Timing Window
timing_window_df = []
for t, trs in enumerate(blocks):
    timing_window =  processed_skittles_data["timingWindow"][trs]
    timing_window_agg = timing_window.mean(axis = 1).reset_index()
    timing_window_agg["median_timingWindow"] = timing_window.median(axis = 1).values
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
tapping_subjects = set(processed_tapping_data.subject) # 303 subjects
skittles_subjects = set(skittles_data_agg.subject) # 385 subjects
tapping_and_skittles_subjects = tapping_subjects & skittles_subjects # 256 subjects

## Filter Down to Overlap
processed_tapping_data = processed_tapping_data.loc[processed_tapping_data.subject.isin(tapping_and_skittles_subjects)]
processed_tapping_data = processed_tapping_data.reset_index(drop=True).copy()
skittles_data_agg = skittles_data_agg.loc[skittles_data_agg.subject.isin(tapping_and_skittles_subjects)]
skittles_data_agg = skittles_data_agg.reset_index(drop = True).copy()

## Age/Gender/Musical Experience Distribution
age_gender_dist = processed_tapping_data.groupby(["age_bin","gender","musical_experience"]).size()

##############################
### Correlation Analysis
##############################

## Summary Matrices
skittles_X = pd.pivot_table(pd.melt(skittles_data_agg.drop(["post_hits"],axis=1),
                                    id_vars=["subject","block"]),
                            index = "subject",
                            columns = ["variable","block"])["value"]
tapping_Y = pd.pivot_table(pd.melt(processed_tapping_data[["subject","qvc","drift","error","trial_speed","condition"]],
                                   id_vars = ["subject","trial_speed","condition"]),
                           index = "subject",
                           columns =["variable","trial_speed","condition"])["value"]

## Keep Mean only (Not Median)
X_to_drop = [c for c in skittles_X.columns.tolist() if "median" in c[0]]
skittles_X = skittles_X.drop(X_to_drop, axis=1)

## Unify Column Names
tapping_Y.columns = ["-".join(list(map(str, c))) for c in tapping_Y.columns.tolist()]
skittles_X.columns = ["-".join(list(map(str,c))) for c in skittles_X.columns.tolist()]

tapping_Y = pd.merge(tapping_Y,
                     processed_tapping_data.set_index("subject")[["preferred_period"]].drop_duplicates(),
                     left_index = True,
                     right_index = True)

## Merge
data_merged = pd.merge(skittles_X, tapping_Y, left_index = True, right_index = True)

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
            annot_kws={"size": 6})
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

## Plot Top Correlations
n_top = 20
top_correlations = corr.loc[tapping_Y.columns, skittles_X.columns].unstack().map(np.abs).reset_index().sort_values(0,ascending = False)[["level_0","level_1"]].values[:n_top]
for skit, tap in top_correlations:

    fig, ax = plt.subplots(1, 2, figsize = (12, 8))
    data_merged.plot.scatter(skit, tap, ax = ax[0])
    data_merged.rank(axis=0).plot.scatter(skit, tap, ax = ax[1])
    pearson_corr = pearsonr(data_merged[skit], data_merged[tap])
    spearman_corr = spearmanr(data_merged[skit], data_merged[tap])
    ax[0].set_title("Pearson Correlation: {:,.2f} (p={:,.2f})".format(pearson_corr[0], pearson_corr[1]))
    ax[1].set_title("Spearman Rank Correlation: {:,.2f} (p={:,.2f})".format(spearman_corr[0], spearman_corr[1]))
    ax[1].set_xlabel("{} subject rank".format(skit))
    ax[1].set_ylabel("{} subject rank".format(tap))
    fig.tight_layout()
    fig.savefig(corr_dir + "{}--{}".format(skit, tap) + FIGURE_FMT)
    fig.savefig(corr_dir + "{}--{}".format(skit, tap) + ".png")
    plt.close()
