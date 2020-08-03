
"""
Exploration of the correlation between Skittles and Tapping performance
"""

FIGURE_FMT = ".pdf"
standard_fig = (6.5, 4.2)

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


def bootstrap_sample(X,
                     Y=None,
                     func=np.nanmean,
                     axis=0,
                     sample_percent=70,
                     samples=100):
    """

    """
    sample_size = int(sample_percent / 100 * X.shape[0])
    if Y is not None:
        sample_size_Y = int(sample_percent / 100 * Y.shape[0])
    estimates = []
    for sample in range(samples):
        sample_ind = np.random.choice(X.shape[0], size=sample_size, replace=True)
        X_sample = X[sample_ind]
        if Y is not None:
            samply_ind_y = np.random.choice(Y.shape[0], size=sample_size_Y, replace=True)
            Y_sample = Y[samply_ind_y]
            sample_est = func(X_sample, Y_sample)
        else:
            sample_est = func(X_sample, axis=axis)
        estimates.append(sample_est)
    estimates = np.vstack(estimates)
    ci = np.percentile(estimates, [2.5, 50, 97.5], axis=axis)
    return ci

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
### Skittles Task Visualizations
##############################

"""
Replicate Visuals Generated by Se-Woong in Matlab

1. Learning Curve: Timing Window Over Trials ("timingWindow")
2. Histogram: Timing Window Per Subject (per Block)
3. Learning Curve: QVC_ITI Over Trials (Smoothed) ("qvcITImovingWindow")
4. Histogram: QVC_ITI Per Subject (Per Block)
5. Learning Curve: Timing Error Per Over Trials ("timingError")
6. Histogram: Timing Error Per Subject (per Block)
"""

## Overlap Subject List
overlap_subject_list = sorted(tapping_and_skittles_subjects)

## Timing Window Learning Curve
timing_window_CI = bootstrap_sample(processed_skittles_data["timingWindow"].loc[overlap_subject_list].values,
                 axis=0,
                 func=np.nanmean)
fig, ax = plt.subplots(figsize=standard_fig)
ax.fill_between(np.arange(1,101),
                timing_window_CI[0],
                timing_window_CI[2],
                color = "navy",
                alpha = 0.3,
                label="95% C.I.")
ax.plot(np.arange(1, 101),
        timing_window_CI[1],
        color = "navy",
        alpha = 0.8,
        linewidth = 2,
        label="Mean")
for i in [25, 50, 75]:
    ax.axvline(i+.5, color="black", linestyle="--", alpha=0.25, zorder=-1)
ax.set_xlabel("Throw Number", fontweight="bold", fontsize = 16)
ax.set_ylabel("Timing Window (ms)", fontweight="bold", fontsize=16)
ax.legend(loc="lower right", frameon=True, fontsize=14, edgecolor="gray", borderpad=0.25, handlelength=2)
ax.tick_params(labelsize=14)
ax.set_title("Learning Curve: Timing Window", fontweight="bold", fontsize=18, loc="left", fontstyle="italic")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("./plots/intertask/learning_curve_timing_window.png", dpi=300)
fig.savefig(f"./plots/intertask/learning_curve_timing_window{FIGURE_FMT}", dpi=300)
plt.close(fig)

## Timing Window Histogram
fig, ax = plt.subplots(4, 1, figsize=standard_fig, sharex=True)
bins = np.linspace(processed_skittles_data["timingWindow"].loc[overlap_subject_list].min(axis=1).min(),
                   np.percentile(processed_skittles_data["timingWindow"].loc[overlap_subject_list].max(axis=1),75),
                   41)
for b, binds in enumerate(blocks):
    block_vals = processed_skittles_data["timingWindow"].loc[overlap_subject_list][binds]
    block_means = block_vals.mean(axis=1)
    bci = bootstrap_sample(block_means.values, func=np.nanmean).T[0]
    ax[b].hist(block_means, bins=bins, color="navy", alpha=0.7, edgecolor="black",
               label="$\\mu={:.1f}$\n95% C.I.=[{:.1f},{:.1f}]".format( bci[1], bci[0], bci[2]))
    ax[b].set_title("Block {}".format(b+1), fontsize=16, loc="left")
    ax[b].spines["top"].set_visible(False)
    ax[b].spines["right"].set_visible(False)
    ax[b].tick_params(labelsize=14)
    ax[b].legend(loc="center right", frameon=False, fontsize=12)
ax[-1].set_xlabel("Mean Timing Window (ms)", fontweight="bold", fontsize=16)
fig.text(0.04, 0.5, "# Subjects", fontweight="bold", fontsize=16, rotation=90, va="center", ha="center")
fig.suptitle("Timing Window Distribution Over Time", fontsize=18, fontweight="bold", y=0.97, fontstyle="italic")
fig.tight_layout()
fig.subplots_adjust(left=0.12, top=.84)
fig.savefig("./plots/intertask/histogram_timing_window.png", dpi=300)
fig.savefig(f"./plots/intertask/histogram_curve_timing_window{FIGURE_FMT}", dpi=300)
plt.close(fig)

## Learning Curve QVC_ITI
qvc_iti_CI = bootstrap_sample(processed_skittles_data["qvcITImovingWindow"].loc[overlap_subject_list].values,
                 axis=0,
                 func=np.nanmean) * 1000
fig, ax = plt.subplots(figsize=standard_fig)
ax.fill_between(np.arange(13,89),
                qvc_iti_CI[0],
                qvc_iti_CI[2],
                color = "navy",
                alpha = 0.3,
                label="95% C.I.")
ax.plot(np.arange(13,89),
        qvc_iti_CI[1],
        color = "navy",
        alpha = 0.8,
        linewidth = 2,
        label="Mean")
for i in [25, 50, 75]:
    ax.axvline(i+.5, color="black", linestyle="--", alpha=0.25, zorder=-1)
ax.set_xlabel("Throw Number", fontweight="bold", fontsize = 16)
ax.set_ylabel("QVC IRI (ms)", fontweight="bold", fontsize=16)
ax.tick_params(labelsize=14)
ax.legend(loc="upper right", frameon=True, fontsize=14, edgecolor="gray", borderpad=0.25, handlelength=2)
ax.set_title("Learning Curve: QVC IRI", fontweight="bold", fontsize=18, loc="left", fontstyle="italic")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlim(0, 100)
fig.tight_layout()
fig.savefig("./plots/intertask/learning_curve_qvc_iri.png", dpi=300)
fig.savefig(f"./plots/intertask/learning_curve_qvc_iri{FIGURE_FMT}", dpi=300)
plt.close(fig)

## QVC Histogram
fig, ax = plt.subplots(1, 1, figsize=standard_fig, sharex=True)
bins = np.linspace(processed_skittles_data["qvcITImovingWindow"].loc[overlap_subject_list][[0,75]].min(axis=1).min(),
                   processed_skittles_data["qvcITImovingWindow"].loc[overlap_subject_list][[0,75]].max(axis=1).max(),
                   41) * 1000
for i, (ind, label) in enumerate(zip([0, 75],["Start","Finish"])):
    vals = processed_skittles_data["qvcITImovingWindow"].loc[overlap_subject_list][ind] * 1000
    bci = bootstrap_sample(vals.values, func=np.nanmean).T[0] 
    ax.hist(vals.values,
            bins=bins,
            color="navy" if i == 1 else "orange",
            alpha=.5,
            edgecolor="navy" if i == 1 else "darkorange",
            label="{} $\\mu={:.1f}$\n95% C.I.=[{:.1f},{:.1f}]".format(label, bci[1], bci[0], bci[2]))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=14)
ax.legend(loc="upper right", frameon=True, fontsize=12, edgecolor="gray", borderpad=0.25, handlelength=2)
ax.set_xlabel("QVC IRI (ms)", fontweight="bold", fontsize=16)
fig.text(0.04, 0.5, "# Subjects", fontweight="bold", fontsize=16, rotation=90, va="center", ha="center")
ax.set_title("QVC IRI Distribution Over Time", fontsize=18, fontweight="bold", fontstyle="italic", loc="left")
ax.set_xlim(left=0)
fig.tight_layout()
fig.subplots_adjust(left=0.12, top=.89)
fig.savefig("./plots/intertask/histogram_qvc_iri.png", dpi=300)
fig.savefig(f"./plots/intertask/histogram_qvc_iri{FIGURE_FMT}", dpi=300)
plt.close(fig)

## Timing Window Learning Curve
timing_error_CI = bootstrap_sample(processed_skittles_data["timingError"].loc[overlap_subject_list].values,
                 axis=0,
                 func=np.nanmean)
fig, ax = plt.subplots(figsize=standard_fig)
ax.fill_between(np.arange(1,101),
                timing_error_CI[0],
                timing_error_CI[2],
                color = "navy",
                alpha = 0.3,
                label="95% C.I.")
ax.plot(np.arange(1, 101),
        timing_error_CI[1],
        color = "navy",
        alpha = 0.8,
        linewidth = 2,
        label="Mean")
for i in [25, 50, 75]:
    ax.axvline(i+.5, color="black", linestyle="--", alpha=0.25, zorder=-1)
ax.set_xlabel("Throw Number", fontweight="bold", fontsize = 16)
ax.set_ylabel("Timing Error (ms)", fontweight="bold", fontsize=16)
ax.legend(loc="upper right", frameon=True, fontsize=14, edgecolor="gray", handlelength=2, borderpad=0.25)
ax.tick_params(labelsize=14)
ax.set_title("Learning Curve: Timing Error", fontweight="bold", fontsize=18, loc="left", fontstyle="italic")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
fig.savefig("./plots/intertask/learning_curve_timing_error.png", dpi=300)
fig.savefig(f"./plots/intertask/learning_curve_timing_error{FIGURE_FMT}", dpi=300)
plt.close(fig)

## Timing Error Histogram
fig, ax = plt.subplots(4, 1, figsize=standard_fig, sharex=True)
bins = np.linspace(processed_skittles_data["timingError"].loc[overlap_subject_list].min(axis=1).min(),
                   np.percentile(processed_skittles_data["timingError"].loc[overlap_subject_list].max(axis=1),75),
                   41)
for b, binds in enumerate(blocks):
    block_vals = processed_skittles_data["timingError"].loc[overlap_subject_list][binds]
    block_means = block_vals.mean(axis=1)
    bci = bootstrap_sample(block_means.values, func=np.nanmean).T[0]
    ax[b].hist(block_means, bins=bins, color="navy", alpha=0.7, edgecolor="navy",
               label="$\\mu={:.1f}$\n95% C.I.=[{:.1f},{:.1f}]".format( bci[1], bci[0], bci[2]))
    ax[b].set_title("Block {}".format(b+1), fontsize=16, loc="left")
    ax[b].spines["top"].set_visible(False)
    ax[b].spines["right"].set_visible(False)
    ax[b].tick_params(labelsize=14)
    ax[b].legend(loc="center right", frameon=False, fontsize=12)
ax[-1].set_xlabel("Mean Timing Error (ms)", fontweight="bold", fontsize=16)
fig.text(0.04, 0.5, "# Subjects", fontweight="bold", fontsize=16, rotation=90, va="center", ha="center")
fig.suptitle("Timing Error Distribution Over Time", fontsize=18, fontweight="bold", y=0.97, fontstyle="italic")
fig.tight_layout()
fig.subplots_adjust(left=0.12, top=.84)
fig.savefig("./plots/intertask/histogram_timing_error.png", dpi=300)
fig.savefig(f"./plots/intertask/histogram_curve_timing_error{FIGURE_FMT}", dpi=300)
plt.close(fig)

## Combined Learning Curves
fig, axes = plt.subplots(2, 2, figsize=standard_fig, sharex=False, sharey=False)
axes = axes.ravel()
for v, (val, name, met) in enumerate(zip(["timingError","timingWindow", "qvcITImovingWindow","Error"],
                                    ["Timing Error", "Timing Window", "QVC IRI", "Distance Error"],
                                    ["Time\n(ms)", "Time\n(ms)", "Time\n(ms)", "Angle\n(radians)"])):
    CI = bootstrap_sample(processed_skittles_data[val].loc[overlap_subject_list].values,
                    axis=0,
                    func=np.nanmean)
    if val == "qvcITImovingWindow":
        CI *= 1000
    ax = axes[v]
    if val == "qvcITImovingWindow":
        ind = np.arange(13, 89)
    else:
        ind = np.arange(1,101)
    ax.fill_between(ind,
                    CI[0],
                    CI[2],
                    color = "navy",
                    alpha = 0.3,
                    label="95% C.I.")
    ax.plot(ind,
            CI[1],
            color = "navy",
            alpha = 0.8,
            linewidth = 2,
            label="Mean")
    for i in [25, 50, 75]:
        ax.axvline(i+.5, color="black", linestyle="--", alpha=0.25, zorder=-1)
    if val == "qvcITImovingWindow":
        ax.set_yticks([70, 90])
    ax.set_ylabel(met, fontweight="bold", fontsize=16, labelpad=8)
    ax.set_title(name, fontweight="bold", style="italic", loc="center", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, 100)
for i in range(2):
    axes[i].set_xticks([])
fig.text(0.01, .94, "(A)", ha="left", va="center", fontweight="bold", fontsize=16, fontstyle="italic")
fig.text(0.5, .94, "(B)", ha="left", va="center", fontweight="bold", fontsize=16, fontstyle="italic")
fig.text(0.01, .525, "(C)", ha="left", va="center", fontweight="bold", fontsize=16, fontstyle="italic")
fig.text(0.5, .525, "(D)", ha="left", va="center", fontweight="bold", fontsize=16, fontstyle="italic")
axes[2].set_xlabel("Throw #", fontweight="bold", fontsize=16)
axes[3].set_xlabel("Throw #", fontweight="bold", fontsize=16)
fig.tight_layout()
fig.subplots_adjust(hspace=.28, wspace=.6)
fig.savefig("./plots/intertask/learning_curves_combined.png", dpi=300)
fig.savefig(f"./plots/intertask/learning_curves_combined{FIGURE_FMT}", dpi=300)
plt.close(fig)

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
    for t, (trial_speed, tlbl) in enumerate(zip(["SpedUp","NoChange","SlowedDown"],["20%\nFaster","Preferred","20%\nSlower"])):
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
                            edgecolor = "navy",
                            linewidth = .5,
                            alpha = .3,
                            color = "navy")
            plot_ax.spines['right'].set_visible(False)
            plot_ax.spines['top'].set_visible(False)
            if plot_rank:
                plot_ax.set_xlim(min(x) - 3, max(x) + 15)
                plot_ax.set_ylim(min(y) - 3, max(y) + 10)
            if t == 2:
                plot_ax.set_xlabel(sklbl, fontsize = 12)
            else:
                if not plot_rank:
                    plot_ax.set_xticks([])
        ax[t][0].set_ylabel(tlbl, fontsize = 12, labelpad = 5)
    fig.text(0.55,
            0.02,
            "Throwing Skill" + {True:" (Rank)",False:""}[plot_rank],
            fontweight = "bold",
            horizontalalignment="center",
            fontsize = 16)
    fig.text(0.035,
             0.55,
             "Tapping Skill{}".format({True:" (Rank)",False:""}[plot_rank]),
             fontweight = "bold",
             horizontalalignment = "center",
             verticalalignment = "center",
             rotation = 90,
             fontsize = 16)
    fig.suptitle("Tapping Metric: {}".format(tap_met_name),
                fontsize = 16,
                x = 0.2,
                y = .95,
                fontweight = "bold",
                horizontalalignment = "left",
                verticalalignment = "center",
                fontstyle = "italic")
    fig.tight_layout()
    fig.subplots_adjust(bottom = 0.2, left = 0.2, hspace = 0.10, top = .92)
    return fig, ax

## Run Plotting
tap_metrics = [("drift","Drift"),("qvc","QVC ITI"),("error","Timing Error")]
for tap_met, met_name in tap_metrics:
    fig, ax = plot_comparison(tap_met, met_name, plot_rank = True)
    fig.savefig("./plots/intertask/{}_correlations".format(tap_met) + FIGURE_FMT)
    fig.savefig("./plots/intertask/{}_correlations".format(tap_met) + ".png")
    plt.close()

##############################
### Miscellaneous Visuals
##############################

## Format
data_merged_ols["mean_iti_seconds"] = data_merged_ols["mean_ITI"] / 1000

## Preferred Period/Mean ITI
pp_iri_corr =  spearmanr(data_merged_ols["preferred_period"], data_merged_ols["mean_ITI"])
lfit = np.polyfit(data_merged_ols["preferred_period"], data_merged_ols["mean_iti_seconds"], 1)

## Preferred Period vs Inter-Release-Interval
fig, ax = plt.subplots(figsize = standard_fig)
data_merged_ols.plot.scatter("preferred_period",
                             "mean_iti_seconds",
                             ax = ax,
                             color = "navy",
                             s = 40,
                             alpha = .3,
                             edgecolor = "navy",
                             )
x = np.linspace(data_merged_ols["preferred_period"].min()*.9, data_merged_ols["preferred_period"].max()*1.1)
ax.plot(x,
        np.polyval(lfit, x),
        label = "Spearman $R={:.2f}$ ($p={:.2f}$)".format(pp_iri_corr[0], pp_iri_corr[1]),
        color = "crimson",
        linestyle = "--",
        alpha = .9,
        linewidth = 2)
ax.set_xlabel("Preferred Period (ms)",
              fontsize = 16,
              fontweight = "bold",
              labelpad = 10)
ax.set_ylabel("Inter-Release-Interval (s)",
              fontsize = 16,
              fontweight = "bold",
              labelpad = 10)
ax.tick_params(labelsize = 14)
ax.set_title("Correlation Plot: Preferred Rhythm", fontweight="bold", fontstyle="italic", fontsize=18, loc="left")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend(loc = "upper right", frameon=True, edgecolor = "gray", handlelength=2, borderpad=.25)
fig.tight_layout()
fig.savefig("./plots/intertask/preferred_period_IRI_scatter" + FIGURE_FMT)
fig.savefig("./plots/intertask/preferred_period_IRI_scatter" + ".png")
plt.close()

##############################
### OLS ANOVA for Rythmicity
##############################

## Fit OLS Model
formula = "qvc_ITI ~ age + C(gender) + C(musical_experience) + C(sport_experience)"
ols_model = smf.ols(formula, data_merged_ols).fit()
aov_results = anova.anova_lm(ols_model)