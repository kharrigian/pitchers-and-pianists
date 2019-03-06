
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
rcParams['font.sans-serif'] = ['Helvetica']
rcParams["errorbar.capsize"] = 5

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
merged_results_df = pd.read_csv("./data/merged_processed_results.csv")

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
def calc_pp(tap_inits, sample_rate = 2000):
    """

    """
    if tap_inits is None or len(tap_inits) < 3:
        return None, 0
    tap_mask = np.logical_and(tap_inits >= 5 * sample_rate, tap_inits <= 15 * sample_rate)
    tap_mask = np.nonzero(tap_mask)[0]
    if len(tap_mask) == 1:
        return None, 0
    tap_diffs = np.diff(tap_inits[tap_mask])
    pp = np.mean(tap_diffs) / sample_rate
    return pp, len(tap_mask)
pp_online["preferred_period_offline"] = pp_online.subject.map(lambda i: calc_pp(processed_taps[i]["tap_initiations"])[0])
pp_online["offline_iti_count"] = pp_online.subject.map(lambda i: calc_pp(processed_taps[i]["tap_initiations"])[1])

## Add flag to note whether the subject was kept in the final analysis
pp_online["subject_kept"] = pp_online.subject.isin(merged_results_df.subject)

## Plot Correlation
fig, ax = plt.subplots()
pp_online.loc[pp_online.subject_kept].plot.scatter("preferred_period_online",
                                                   "preferred_period_offline",
                                                   ax = ax,
                                                   label = "Kept",
                                                   color = "C0")
pp_online.loc[~pp_online.subject_kept].plot.scatter("preferred_period_online",
                                                    "preferred_period_offline",
                                                    ax = ax,
                                                    label = "Discared",
                                                    color = "C1")
plt.plot([ax.get_xlim()[0], ax.get_xlim()[1]],
         [ax.get_xlim()[0], ax.get_xlim()[1]],
         color = "black",
         alpha = .3,
         linestyle = "--")
plt.xlabel("Online Calculation (s)")
plt.ylabel("Offline Calculation (s)")
plt.legend(loc = "upper left")
plt.show()

## Correlation
print(pp_online.loc[pp_online.subject_kept].drop(["subject","subject_kept","offline_iti_count"],axis=1).corr())

## Number of Recognized ITIs vs. PP
pp_online.plot.scatter("offline_iti_count", "preferred_period_offline")
plt.show()
