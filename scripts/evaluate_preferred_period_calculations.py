
#### Evaluate the Accuracy of the preferred period calculations

##############################
### Imports
###############################

# Warning Supression
import warnings
warnings.simplefilter("ignore")

# Standard I/O and Data Handling
import pandas as pd
import numpy as np
import os, glob, sys
import datetime
import copy
import pickle

# Data Loading
from scripts.libraries.helpers import load_pickle, dump_pickle, load_tapping_data

# Detection
from scripts.libraries.tap_detection import find_taps

# Plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams

###############################
###  Plot Helpers
###############################

## Plotting Variables
standard_fig = (10,5.8)
plot_dir = "./plots/"
rcParams["errorbar.capsize"] = 5

###############################
###  General Helpers
###############################

## Bootstrapped Confidence Interval
def bootstrap_ci(values,
                 alpha = 0.05,
                 func = np.mean,
                 sample_percent = 70,
                 samples = 100,
                 replace = False):
    processed_vals = []
    values = np.array(values)
    for sample in range(samples):
        sample_vals = np.random.choice(values, int(values.shape[0] * sample_percent/100.), replace = replace)
        processed_vals.append(func(sample_vals))
    return np.percentile(processed_vals, [alpha*100/2, 50, 100. - (alpha*100/2)])

###############################
### Load Data
###############################

# Main Data Directory
data_dir = "./data/"

# Tapping Data Directory
tapping_data_dir = data_dir + "tapping/"

# Tapping Filenames
tapping_filenames = glob.glob(tapping_data_dir + "*/*")
tapping_filenames = [tapping_filenames[i] for i in np.argsort([int(t.split("/")[-1].replace(".mat","")) for t in tapping_filenames])]

# Load all data into memory
tapping_data = {}
for subject_file in tapping_filenames:
    ## Extract subject ID from the filename
    subject_id = int(subject_file.split("/")[-1].replace(".mat",""))
    ## Load data and add to cache
    subject_data = load_tapping_data(subject_file)
    ## Get Preferred Period Calibration Signal and Estimated PP
    pref_force = subject_data["preferred_force"]
    pref_period_calc = subject_data["preferred_period_online"]
    ## Cache
    tapping_data[subject_id] = {"data":pref_force, "preferred_period":pref_period_calc}

## Load Merged Results to See Which Subjects Were Kept
results_df = pd.read_csv("./data/processed_results.csv")

###############################
### Calculations
###############################

# Store for processed taps
cache_file = "./data/preferred_periods.pickle"
processed_taps = {}
if os.path.exists(cache_file):
    processed_taps = load_pickle(cache_file)

# Process
for subject, data in tapping_data.items():

    if subject in processed_taps:
        continue

    ## Estimate Taps
    est_taps = find_taps(data["data"],
                        expected_intertapinterval = 1000)

    # Note the starting number of taps
    starting_tap_init_length = len(est_taps)

    # Interactive Removal Procedure
    filtering_complete = False
    while not filtering_complete:

        # Define function to manually remove identified taps
        def onclick(event):
            global ix, iy
            ix, iy = event.xdata, event.ydata
            # assign global variable to access outside of function
            global coords
            coords.append((ix, iy))
            # Disconnect after 2 clicks
            if len(coords) == 2:
                fig.canvas.mpl_disconnect(cid)
                plt.close(1)

        # Separate by condition and compute ITIs
        itis = np.diff(est_taps)

        # Create Plot
        fig, ax = plt.subplots(2,1, sharex = False, figsize = (14,6))
        ax[0].plot(data["data"])
        ax[0].vlines(est_taps, ax[0].get_ylim()[0]-0.05, ax[0].get_ylim()[1],
                    color = "red", linewidth = .75, linestyle = "--")
        t1 = ax[1].plot(np.arange(len(itis)), itis)
        t2 = ax[1].scatter(np.arange(len(itis)), itis, color = "red", s = 20, marker = "o")
        ax[1].axvline(len(itis) - .5, color = "black", linestyle = "--")
        fig.tight_layout()

        # Interact with Plot and Store Chosen Coordinates
        coords = []
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show(1)

        # Remove taps within selected interval
        start_len, end_len = 0, 0
        if len(coords) > 1:
            cmin, cmax = coords[0][0], coords[1][0]
            start_len = len(est_taps)
            est_taps = est_taps[np.logical_not((est_taps >= cmin ) & (est_taps <= cmax))]
            end_len = len(est_taps)
        else:
            filtering_complete = True
            continue

        # Check for continuation
        if start_len == end_len:
            filtering_complete = True
        else:
            keep_filtering = input("Continue filtering? ")
            if len(keep_filtering) == 0:
                filtering_complete = True

    # Discard Check
    final_check = ""
    filtered_original = False
    if starting_tap_init_length != len(est_taps):

        # Flag to note that taps were filtered
        filtered_original = True

        # Show Final Result
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(data["data"])
        ax.vlines(est_taps, ax.get_ylim()[0], ax.get_ylim()[1], color = "red", linewidth = .75, linestyle = "--")
        fig.tight_layout()
        plt.show(block=False)

        # Ask user to validate final filtering
        final_check = input("Discard? If yes, why? ")

        # Double check plot is closed
        plt.close("all")

    # Save results
    discard_reason = None
    if len(final_check) > 0:
        discard_reason = final_check
        est_taps = None
    processed_taps[subject] = {"tap_initiations":est_taps, "discard_reason":discard_reason, "filtered":filtered_original}

## Dump
dump_pickle(processed_taps, cache_file)

###############################
### Comparisons
###############################

## Extract Preferred Periods
pp_online = pd.DataFrame([[sub, data["preferred_period"]] for sub, data in tapping_data.items()],
                         columns = ["subject","preferred_period_online"])

## Calculate PP from Tap Initiations
def calc_pp(tap_inits,
            sample_rate = 2000,
            **bootstrap_kwargs):
    """

    """
    if tap_inits is None or len(tap_inits) < 3:
        return None, 0, (None,None,None)
    tap_mask = np.logical_and(tap_inits >= 5 * sample_rate, tap_inits <= 15 * sample_rate)
    tap_mask = np.nonzero(tap_mask)[0]
    if len(tap_mask) == 1:
        return None, 0, (None,None,None)
    tap_diffs = np.diff(tap_inits[tap_mask])
    pp = np.mean(tap_diffs) / sample_rate
    conf_int = bootstrap_ci(tap_diffs / sample_rate, **bootstrap_kwargs)
    return pp, len(tap_mask), tuple(conf_int)

## Execute Calculation
pp_online["pp_stats"] = pp_online.subject.map(lambda i: calc_pp(processed_taps[i]["tap_initiations"]))
for c, col in enumerate(["preferred_period_offline","offline_iti_count","pp_confidence_interval"]):
    pp_online[col] = pp_online["pp_stats"].map(lambda i: i[c])
pp_online.drop(["pp_stats"],axis=1,inplace=True)

## Add flag to note whether the subject was kept in the final analysis
pp_online["subject_kept"] = pp_online.subject.isin(results_df.subject)

## Correlation
kept_corr = pp_online.loc[pp_online.subject_kept].drop(["subject","subject_kept","offline_iti_count","pp_confidence_interval"],axis=1).corr()
discarded_corr = pp_online.loc[~pp_online.subject_kept].drop(["subject","subject_kept","offline_iti_count","pp_confidence_interval"],axis=1).corr()

## Difference
pp_online["calculation_difference"] = pp_online["preferred_period_online"] - pp_online["preferred_period_offline"]

## Plot Correlation
fig, ax = plt.subplots()
pp_online.loc[pp_online.subject_kept].plot.scatter("preferred_period_online",
                                                   "preferred_period_offline",
                                                   ax = ax,
                                                   label = "Kept (Pearson $r = {:,.3f}$)".format(kept_corr.values[1,0]),
                                                   color = "C0")
pp_online.loc[~pp_online.subject_kept].plot.scatter("preferred_period_online",
                                                    "preferred_period_offline",
                                                    ax = ax,
                                                    label = "Discarded (Pearson $r = {:,.3f}$)".format(discarded_corr.values[1,0]),
                                                    color = "C1")
plt.plot([ax.get_xlim()[0], ax.get_xlim()[1]],
         [ax.get_xlim()[0], ax.get_xlim()[1]],
         color = "black",
         alpha = .3,
         linestyle = "--")
plt.xlabel("Online Calculation (s)")
plt.ylabel("Offline Calculation (s)")
plt.legend(loc = "upper left")
plt.tight_layout()
plt.savefig("./plots/methodology/preferred_period_scatter_comparison.png")
plt.close()

## Histogram of Differences
fig, ax = plt.subplots()
pp_online.calculation_difference.hist(bins = 50, ax = ax, zorder = 5)
ax.set_xlabel("Online Calculation minus Offiline Calculation (s)")
ax.set_ylabel("# of Subjects")
plt.tight_layout()
plt.savefig("./plots/methodology/preferred_period_online_offline_difference_histogram.png")
plt.close()

## Number of Recognized ITIs vs. PP
fig, ax = plt.subplots()
pp_online.loc[pp_online.subject_kept].plot.scatter("offline_iti_count",
                                                   "preferred_period_offline",
                                                   ax=ax,
                                                   label = "Kept",
                                                   color = "C0")
pp_online.loc[~pp_online.subject_kept].plot.scatter("offline_iti_count",
                                                    "preferred_period_offline",
                                                    ax=ax,
                                                    label = "Discarded",
                                                    color = "C1")

plt.show()

## Calculate Whether Subject's Online Period Fell within Bootstrapped Range
in_range = lambda row: row["preferred_period_online"] <= row["pp_confidence_interval"][2] and \
                       row["preferred_period_online"] >= row["pp_confidence_interval"][0] if \
                       (not pd.isnull(row["preferred_period_online"]) and \
                        not any(i is None for i in row["pp_confidence_interval"])) else \
                       False
pp_online["online_in_range"] = pp_online.apply(in_range, axis = 1)

## Within Variation Range (Preferred Period Confidence Interval)
conf_ints = pp_online.sort_values("preferred_period_online").dropna().pp_confidence_interval.values
online_calc = pp_online.sort_values("preferred_period_online").dropna().preferred_period_online.values
fig, ax = plt.subplots()
for i, (val_conf, val_online) in enumerate(zip(conf_ints, online_calc)):
    ax.vlines(i, val_conf[0], val_conf[2], color = "C0")
    ax.scatter(i,
               val_online,
               color = "black" if val_online <= val_conf[2] and val_online >= val_conf[0] else "red",
               s = 10,
               zorder = 10)
plt.show()

###############################
### Subject-Filtering
###############################

"""
Subjects whom were given trials based on a preferred period online
calculation that differed substantially from the offline calculation
are flagged for removal here (since their data might not represent
normal behavior)
"""

## Calculate Absolute Difference (Relative to Offline Calculation)
pp_online["rel_difference"] = (pp_online["calculation_difference"] / pp_online["preferred_period_offline"]) * 100
pp_online["absolute_rel_difference"] = np.abs(pp_online["rel_difference"])

## Difference Cumulative Distribution
axes = pp_online.hist("absolute_rel_difference",
                       by = "subject_kept",
                       cumulative = True,
                       bins = list(np.arange(0,101)) + [pp_online.absolute_rel_difference.max() +1],
                       histtype = "step")
axes[0].set_title("Subjects Discarded Already")
axes[1].set_title("Subjects Currently Kept")
for a in axes:
    a.set_xlabel("Relative Difference Threshold")
    a.set_ylabel("# Subjects After Removal")
plt.show()

## Boundaries
def show_boundaries_at_difference_threshold(threshold,
                                            show = True):
    """
    Create a scatter plot of online/offline preferred period calculations. Show
    filtering boundary and compute the number of subjects removed under a given
    threshold

    Args:
        threshold (numeric): Percentage (100 scale) to use for relative absolute difference
        show (bool): If True, show the plot. Otherwise, return the figure

    Returns:
        None or fig, ax combo (depending on the `show` parameter)
    """
    fig, ax = plt.subplots()
    ax.scatter(pp_online.loc[pp_online.subject_kept]["preferred_period_offline"],
               pp_online.loc[pp_online.subject_kept]["preferred_period_online"],
               s = 50,
               alpha = .3,
               color = "C0",
               label = "Currently Kept")
    ax.scatter(pp_online.loc[~pp_online.subject_kept]["preferred_period_offline"],
               pp_online.loc[~pp_online.subject_kept]["preferred_period_online"],
               s = 50,
               alpha = .3,
               color = "C1",
               label = "Currently Discarded")
    xlim = list(ax.get_xlim())
    ax.plot(xlim,
            xlim,
            color = "black",
            alpha = .3,
            linestyle = "-",
            label = "Match")
    ax.plot(xlim,
            np.array(xlim) *  (100 + threshold) / 100.,
            alpha = .3,
            color = "red",
            linestyle = "--")
    ax.plot(xlim,
            np.array(xlim) *  (100 - threshold) / 100.,
            alpha = .3,
            color = "red",
            linestyle = "--",
            label = "Boundary at {}%".format(threshold))
    ax.legend(loc = "lower right", frameon = True, fontsize = 8)
    rem_tot = (pp_online.absolute_rel_difference >= threshold).sum()
    rem_new = (pp_online.loc[pp_online.subject_kept].absolute_rel_difference >= threshold).sum()
    ax.set_title("{} Subjects Removed ({} new)".format(rem_tot, rem_new))
    ax.set_xlabel("Offline Preferred Period (s)")
    ax.set_ylabel("Online Preferred Period (s)")
    fig.tight_layout()
    if show:
        plt.show()
    else:
        return fig, ax

## Plot Filtering
fig, ax = show_boundaries_at_difference_threshold(5, False)
fig.savefig("./plots/methodology/pp_filtering_boundaries.png")
plt.close()

## Create Filter Table Data Cache for `statistical_testing.py` script
filter_cache_file_out = "./data/preferred_period_filtering_map.csv"
pp_online[["subject","rel_difference","absolute_rel_difference","online_in_range"]].to_csv(filter_cache_file_out,index=False)
