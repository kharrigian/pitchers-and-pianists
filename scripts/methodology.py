
### Visualizations to showcase the methodology

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
method_plots = plot_dir + "methodology/"
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica']
rcParams["errorbar.capsize"] = 5

## Plot subdirectories
for d in [plot_dir, method_plots]:
    if not os.path.exists(d):
        os.mkdir(d)

###############################
### Load Data
###############################

# Main Data Directory
data_dir = "./data/"

# Tapping Data Directory
tapping_data_dir = data_dir + "tapping/"

# Load Tap Data
stage_4_processed_file = data_dir + "stage_4_processed.pickle"
stage_4_processed = load_pickle(stage_4_processed_file)

# Tapping Filenames
tapping_filenames = glob.glob(tapping_data_dir + "*/*")
tapping_filenames = [tapping_filenames[i] for i in np.argsort([int(t.split("/")[-1].replace(".mat","")) for t in tapping_filenames])]

# Load all data into memory
tapping_data = {}
for subject_file in tapping_filenames:
    ## Extract subject ID from the filename
    subject_id = int(subject_file.split("/")[-1].replace(".mat",""))
    ## Locate the processed taps
    subject_taps = stage_4_processed[subject_id]
    ## Ignore subject if we didn't process them successfully
    if subject_taps is None:
        continue
    ## Load data and add to cache
    subject_data = load_tapping_data(subject_file)
    subject_data["processed_taps"] = subject_taps
    tapping_data[subject_id] = subject_data

##########################
### Plotting Functions
##########################

def plot_experimental_design(subject, trial):
    """
    Plot experimental design
    """
    ## Check that subject data was processed
    if subject not in tapping_data:
        raise ValueError("Subject {} does not have any usable data".format(subject))
    ## Identify relevant data
    subject_data = tapping_data[subject]
    preferred_period = float(subject_data["preferred_period_online"]) * 1000
    trial_frequency = subject_data["frequency_sequence"]
    metronome_signal = subject_data["trial_metronome"]
    trial_force = subject_data["trial_force"]
    trial_taps = subject_data["processed_taps"]
    ## Initialize Figure Grid
    fig, ax = plt.subplots(2, 1, figsize = standard_fig, sharex = False)
    ax[0].plot(np.arange(len(trial_force[trial-1]))/2000., trial_force[trial-1], color = "navy",
                linewidth = 2)
    ax[0].axvline(15., linestyle = "--", linewidth = 2, color = "red", label = "Metronome Ends")
    ax[0].set_xlabel("Time (s)", fontsize = 16, labelpad = 5, fontweight = "bold")
    ax[0].set_ylabel("Force", fontsize = 16, labelpad = 5, fontweight = "bold")
    ax[0].set_xlim(0, len(trial_force[trial-1])/2000.)
    ax[0].set_ylim(min(trial_force[trial-1]), max(trial_force[trial-1]) * 1.1)
    sync_taps = trial_taps[trial]["metronome"][-9:][:,0]
    cont_taps = trial_taps[trial]["no_metronome"][-9:][:,0]
    ax[0].fill_between([np.mean(sync_taps[-9:-7])/2000., 15], ax[0].get_ylim()[0] - 1, ax[0].get_ylim()[1] + 1,
                        color = "blue", alpha = .2, label = "Synchronization Window")
    ax[0].fill_between([np.mean(cont_taps[-9:-7])/2000., len(trial_force[trial-1])/2000.], ax[0].get_ylim()[0] - 1, ax[0].get_ylim()[1] + 1,
                        color = "green", alpha = .2, label = "Continuation Window")
    met_ind = np.arange(len(trial_taps[trial]["metronome"]))+1
    nomet_ind = np.arange(max(met_ind)+1, max(met_ind)+1 + len(trial_taps[trial]["no_metronome"]))
    ax[1].plot(met_ind, trial_taps[trial]["metronome"][:,2]/2., color = "navy", linewidth = 2, linestyle = "-")
    ax[1].scatter(met_ind, trial_taps[trial]["metronome"][:,2]/2., s = 50, color = "blue", edgecolor = "navy", zorder = 3)
    ax[1].plot(nomet_ind, trial_taps[trial]["no_metronome"][:,2]/2., color = "navy", linewidth = 2, linestyle = "-")
    ax[1].scatter(nomet_ind, trial_taps[trial]["no_metronome"][:,2]/2., s = 50, color = "blue", edgecolor = "navy", zorder = 3)
    ax[1].axvline(max(met_ind) + .5, linewidth = 2, linestyle = "--", color = "red", label = "Metronome Ends")
    ax[1].axhline(preferred_period, linestyle = "--", color = "black", linewidth = 1.5)
    ax[1].fill_between([min(met_ind[-8:]) - .5, max(met_ind) + .5], ax[1].get_ylim()[0] - 100, ax[1].get_ylim()[1] + 100,
                        color = "blue", alpha = .2, label = "Synchronization Window")
    ax[1].fill_between([min(nomet_ind[-8:]) - .5, max(nomet_ind) + .5], ax[1].get_ylim()[0] - 100, ax[1].get_ylim()[1] + 100,
                        color = "green", alpha = .2, label = "Continuation Window")
    ax[1].text(max(nomet_ind)+1, preferred_period, "Preferred/Trial\nPeriod" if trial_frequency[trial-1] == 1 else "Preferred Period",
                    fontsize = 14, va = "center", multialignment = "center")
    if trial_frequency[trial-1] != 1:
        ax[1].axhline(preferred_period * trial_frequency[trial-1], linestyle = "--", color = "black", linewidth = 1.5)
        ax[1].text(max(nomet_ind) + 1, preferred_period * trial_frequency[trial-1], "Trial Period",  fontsize = 14, va = "center")
    ax[1].set_xlabel("Tap #", fontsize = 16, labelpad = 5, fontweight = "bold")
    ax[1].set_ylabel("Inter Tap\nInterval (ms)", fontsize = 16, labelpad = 5, multialignment = "center", fontweight = "bold")
    ax[1].set_xlim(-.5, max(nomet_ind)+.5)
    met_itis = trial_taps[trial]["metronome"][:,2]
    nomet_itis = trial_taps[trial]["no_metronome"][:,2]
    iti_min, iti_max = min(min(met_itis),min(nomet_itis))/2 * .95 , max(max(met_itis),max(nomet_itis))/2 * 1.05
    trial_freq = trial_frequency[trial-1]
    trial_period = preferred_period * trial_freq
    ymin = iti_min if trial_freq == 1 else min(iti_min, preferred_period * .95) if trial_freq == 1.2 else min(iti_min, trial_period * .95)
    ymax = iti_max if trial_freq == 1 else max(iti_max, preferred_period * 1.05) if trial_freq == 0.8 else max(iti_max, trial_period * 1.05)
    ax[1].set_ylim(ymin, ymax)
    for a in ax:
        a.tick_params(labelsize = 14)
    ax[0].get_yaxis().set_label_coords(-0.1,0.5)
    ax[1].get_yaxis().set_label_coords(-0.1,0.5)
    handles, labels = ax[1].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc = (.075, .935), ncol = 3, fontsize = 14, borderpad = .2, handlelength = 2)
    for t in leg.texts:  t.set_multialignment('center')
    for a in ax:
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.subplots_adjust(right = .825, top = .91)
    return fig, ax

def plot_method_comparison(subject, trial, peakthresh = 50):
    """
    Compare tap detection methods
    """
    ## Check that subject data was processed
    if subject not in tapping_data:
        raise ValueError("Subject {} does not have any usable data".format(subject))
    ## Identify relevant data
    subject_data = tapping_data[subject]
    preferred_period = float(subject_data["preferred_period_online"]) * 1000
    trial_frequency = subject_data["frequency_sequence"][trial-1]
    metronome_signal = subject_data["trial_metronome"][trial-1]
    trial_force = subject_data["trial_force"][trial-1]
    trial_taps = subject_data["processed_taps"][trial]
    ## Get Taps using HMM Method
    taps_hmm = trial_taps["inits"]
    ## Get Taps using Standard Method
    taps_standard = find_taps(trial_force, trial_frequency * preferred_period, use_hmm = False, peakutils_threshold = peakthresh)
    ## Plot Comparison
    fig, ax = plt.subplots(2, 1, figsize = standard_fig, sharex = True)
    time_ind = np.arange(len(trial_force))/2000.
    ax[0].plot(time_ind, trial_force, color = "blue", linewidth = 2)
    ax[0].vlines(taps_hmm / 2000., min(trial_force), max(trial_force) * 1.1, color = "red", linewidth = 1, linestyle = "-")
    ax[1].plot(time_ind, trial_force, color = "blue", linewidth = 2)
    ax[1].vlines(taps_standard / 2000.,min(trial_force), max(trial_force) * 1.1,  color = "red", linewidth = 1, linestyle = "-")
    for a in ax:
        a.tick_params(labelsize = 14)
        a.spines['right'].set_visible(False)
        a.spines['top'].set_visible(False)
        a.set_ylim(min(trial_force), max(trial_force) * 1.1)
        a.set_ylabel("Force", fontsize = 16, labelpad = 5, fontweight = "bold")
        a.set_xlim(0, len(trial_force)/2000.)
    ax[1].set_xlabel("Time (s)", fontsize = 16, labelpad = 5, fontweight = "bold")
    ax[0].set_title("HMM-based Tap Detection", fontsize = 18)
    ax[1].set_title("Peak-based Tap Detection", fontsize = 18)
    fig.tight_layout()
    return fig, ax

def plot_method_comparison_with_zoom(subject, trial, peakthresh = 50, time_start = 15, time_stop = 20):
    """
    Plot detection comparison (with additional zoomed in comparisons)
    """
    ## Check that subject data was processed
    if subject not in tapping_data:
        raise ValueError("Subject {} does not have any usable data".format(subject))
    ## Identify relevant data
    subject_data = tapping_data[subject]
    preferred_period = float(subject_data["preferred_period_online"]) * 1000
    trial_frequency = subject_data["frequency_sequence"][trial-1]
    metronome_signal = subject_data["trial_metronome"][trial-1]
    trial_force = subject_data["trial_force"][trial-1]
    trial_taps = subject_data["processed_taps"][trial]
    ## Get Taps using HMM Method
    taps_hmm = trial_taps["inits"]
    ## Get Taps using Standard Method
    taps_standard = find_taps(trial_force, trial_frequency * preferred_period, use_hmm = False, peakutils_threshold = peakthresh)
    ## Plot Comparison
    time_ind = np.arange(len(trial_force))/2000.
    fig, ax = plt.subplots(2, 2, figsize = standard_fig, sharex = False, sharey = False)
    for col in ax:
        for row in col:
            row.plot(time_ind, trial_force, color = "blue", linewidth = 1, alpha = .5)
            row.set_xlim(0, len(trial_force)/2000.)
            row.set_ylim(min(trial_force), max(trial_force)*1.1)
            row.tick_params(labelsize = 14)
            row.spines['right'].set_visible(False)
            row.spines['top'].set_visible(False)
    ax[0,0].vlines(taps_hmm / 2000., min(trial_force), max(trial_force) * 1.1, color = "red", linewidth = .75, linestyle = ":")
    ax[1,0].vlines(taps_hmm / 2000., min(trial_force), max(trial_force) * 1.1, color = "red", linewidth = 1, linestyle = "--")
    ax[0,1].vlines(taps_standard / 2000.,min(trial_force), max(trial_force) * 1.1,  color = "red", linewidth = .75, linestyle = ":")
    ax[1,1].vlines(taps_standard / 2000.,min(trial_force), max(trial_force) * 1.1,  color = "red", linewidth = 1, linestyle = "--")
    ax[1,0].set_xlim(time_start, time_stop)
    ax[1,1].set_xlim(time_start, time_stop)
    ymin_top = min(trial_force) * .9; ymax_top = max(trial_force) * 1.05
    ymax_bottom = max(trial_force[int(time_start * 2000):int(time_stop * 2000)]) * 1.05
    for i in range(2): ax[1, i].set_ylim(top = ymax_bottom)
    for i in range(2): ax[0, i].set_ylim(ymin_top, ymax_top)
    ax[0,0].set_title("HMM-based Tap Detection", fontsize = 18, fontweight = "bold")
    ax[0,1].set_title("Peak-based Tap Detection", fontsize = 18, fontweight = "bold")
    ax[0,0].set_ylabel("Force", fontsize = 16); ax[1,0].set_ylabel("Force", fontsize = 16)
    ax[1,0].set_xlabel("Time (s)", fontsize = 16); ax[1,1].set_xlabel("Time (s)", fontsize = 16)
    fig.tight_layout()
    return fig, ax

def plot_subject_drift(subject, sharey = True):
    """
    Plot best example of within-subject drift
    """
    ## Get Subject Data
    subject_data = tapping_data[subject]
    preferred_period = float(subject_data["preferred_period_online"]) * 1000
    trial_frequency = subject_data["frequency_sequence"]
    metronome_signal = subject_data["trial_metronome"]
    trial_taps = subject_data["processed_taps"]
    ## Identify Most Exemplary Drift Values
    drift_values = []
    for trial, freq in zip(range(1,7), trial_frequency):
        met_med = np.median(trial_taps[trial]["metronome"][-8:,2])
        nomet_med = np.median(trial_taps[trial]["no_metronome"][-8:,2])
        drift_val = (nomet_med - met_med) / met_med * 100
        drift_values.append((trial, freq, drift_val))
    drift_values = pd.DataFrame(drift_values, columns = ["trial","freq","drift"])
    sped_trial = drift_values.loc[drift_values.freq == 0.8].sort_values("drift", ascending = False)["trial"].values[0]
    nochange_trial = drift_values.loc[drift_values.loc[drift_values.freq == 1.0].drift.map(np.abs).idxmin()]["trial"].astype(int)
    slowed_trial = drift_values.loc[drift_values.freq == 1.2].sort_values("drift", ascending = True)["trial"].values[0]
    ## Create Plot
    fig, ax = plt.subplots(1,3, figsize=(standard_fig[0], standard_fig[1]*.7), sharey=sharey)
    min_seen, max_seen = 10000, 0
    for t, trial in enumerate([sped_trial, nochange_trial, slowed_trial]):
        taps = trial_taps[trial]
        trial_met = taps["metronome"]; met_ind = np.arange(len(trial_met)) + 1
        trial_nomet = taps["no_metronome"]; nomet_ind = np.arange(met_ind.max()+1, met_ind.max()+1 + len(trial_nomet))
        max_iti =  max(max(trial_met[:,2]),max(trial_nomet[:,2]))/2
        min_iti = min(min(trial_met[:,2]),min(trial_nomet[:,2]))/2
        if max_iti > max_seen:
            max_seen = max_iti
        if min_iti < min_seen:
            min_seen = min_iti
    for t, trial in enumerate([sped_trial, nochange_trial, slowed_trial]):
        taps = trial_taps[trial]
        trial_met = taps["metronome"]; met_ind = np.arange(len(trial_met)) + 1
        trial_nomet = taps["no_metronome"]; nomet_ind = np.arange(met_ind.max()+1, met_ind.max()+1 + len(trial_nomet))
        ax[t].plot(met_ind, trial_met[:,2]/2, color = "navy", linewidth = 1, linestyle = "-", alpha = .3)
        ax[t].scatter(met_ind, trial_met[:,2]/2., s = 15, color = "blue", edgecolor = "navy", zorder = 3, alpha = .3)
        ax[t].plot(nomet_ind, trial_nomet[:,2]/2., color = "navy", linewidth = 1, linestyle = "-", alpha = .3)
        ax[t].scatter(nomet_ind, trial_nomet[:,2]/2., s = 15, color = "blue", edgecolor = "navy", zorder = 3, alpha = .3)
        ax[t].axvline(max(met_ind) + .5, linewidth = 2, linestyle = "--", color = "red", label = "Metronome Ends", alpha = .3)
        ax[t].axhline(preferred_period, linestyle = "--", color = "green", linewidth = 2, zorder = 3,
                        label = "Preferred Period" if t == 0 else "", alpha = .3)
        if trial_frequency[trial-1] != 1:
            ax[t].axhline(preferred_period * trial_frequency[trial-1], linestyle = "--", color = "blue", linewidth = 2,
                        zorder = 3, label = "Trial Period" if t == 0 else "", alpha = .3)
        ## drift values
        met_med = np.median(trial_met[-8:,2])/2
        nomet_med = np.median(trial_nomet[-8:,2])/2
        ax[t].fill_between(met_ind[-8:], met_med - 2, met_med + 2, color = "black", alpha = 1)
        ax[t].fill_between(nomet_ind[-8:], nomet_med - 2, nomet_med + 2, color = "black", alpha = 1)
        ax[t].set_xlim(-.5, max(nomet_ind)+.5)
        ax[t].set_xlabel("Tap #", fontsize = 14, labelpad = 5, fontweight = "bold")
        ax[t].tick_params(labelsize = 14)
        ax[t].spines['right'].set_visible(False)
        ax[t].spines['top'].set_visible(False)
        ax[t].fill_between([min(met_ind[-8:]) - .5, max(met_ind) + .5], min_seen - 20, max_seen + 20,
                            color = "blue", alpha = .2, label = "Synchronization Window")
        ax[t].fill_between([min(nomet_ind[-8:]) - .5, max(nomet_ind) + .5], min_seen- 20, max_seen + 20,
                            color = "green", alpha = .2, label = "Continuation Window")
    ax[0].set_ylabel("Inter Tap\nInterval (ms)", fontsize = 14, labelpad = 5, multialignment = "center", fontweight = "bold")
    for j, title in enumerate(["80%\nPreferred Period","100%\nPreferred Period","120%\nPreferred Period"]):
        ax[j].set_title(title, fontsize = 16, fontweight = "bold")
        ax[j].set_ylim(min_seen - 10, max_seen + 10)
    fig.tight_layout()
    return fig, ax

###############################
### Plot Experimental Design
###############################

## Choose Trials to Highlight
design_cases = [(61,2),(69,3),(80,2),(161,2),(179,3),(194,3),(250,3),
                            (258,1),(268,4),(287,5),(307,4),(310,3),(314,3),(314,6),(177,1),
                            (380,5),(431,1),(431,5), (431,3)]

## Plot the Examples
for subject, trial in design_cases:
    fig, ax = plot_experimental_design(subject, trial)
    fig.savefig(method_plots + "{}_{}.png".format(subject, trial),dpi=300)
    plt.close()

###############################
### Plot Method Comparison
###############################

## Choose Cases to Highlight
method_cases = [(110, 5), (76,1), (76,4), (77,1), (80,1), (85,5), (87,1), (87,2), (87,3),
                    (87,6), (88,4), (95,2), (108,1), (109, 1), (109,6), (110, 5),
                    (111, 1), (131, 1), (131,4), (131, 6), (134,1), (135, 1),
                    (163, 6), ]

## Plot the cases
for subject, trial in method_cases:
    fig, ax = plot_method_comparison_with_zoom(subject, trial, peakthresh = 70, time_start = 2, time_stop = 5)
    fig.savefig(method_plots + "comparison_{}_{}.png".format(subject, trial), dpi = 300)
    plt.close()

###############################
### Plot Within-Subject Drift
###############################

## Choose Examples to Highlight
within_subject_examples = [56, 88, 130, 161,177, 225, 249, 380, 382]

## Create Plots
for subject in within_subject_examples:
    fig, ax = plot_subject_drift(subject)
    fig.savefig(method_plots + "drift_{}.png".format(subject), dpi = 300)
    plt.close()

# ###############################
# ### Find good ITI examples
# ###############################
#
# ## Combined Plot for Good Examples
# tuple_sets = [[(88,3), (88,2) ,(88,1)],
#                 [(194,3), (194,1) ,(194,2)],
#                 [(177,5), (177,1) ,(177,6) ],
#                 [(265,4), (265,3) ,(265,2) ],
#                 [(307,4), (307,3) ,(307,2) ],
#                 [(370,6), (370,5) ,(370,4) ],
#                 [(380,3), (380,4) ,(380,5) ],
#                 [(431,5), (431,3) ,(431,1) ],
#                 [(81,5),(81,4),(81,6)],
#                 [(119,5),(119,6),(119,4)],
#                 [(400,6),(400,5),(400,4)],
#                 [(131,5),(131,6),(131,4)],
# ]
# for ts in tuple_sets:
#     sub = ts[0][0]
#     fig, ax = plot_drift_examples(ts, sharey = True)
#     fig.savefig("./plots/methodology/consolidated_drift_{}.png", dpi=300.format(sub))
#     plt.close()
#
# ###############################
# ### Example
# ###############################
#
# ## Trial Methods
# special_methods_cases = [(61,2),(69,3),(80,2),(161,2),(179,3),(194,3),(250,3),
#                             (258,1),(268,4),(287,5),(307,4),(310,3),(314,3),(314,6),(177,1),
#                             (380,5),(431,1),(431,5), (431,3)]
# for subject, trial in special_methods_cases:
#     fig, ax = plot_trial_methods(subject, trial)
#     fig.savefig(method_plots + "{}_{}.png", dpi=300.format(subject, trial))
#     plt.close()
#
# ## Peak estimated_locations
# special_est_cases = [(55,4,4, "lower right"),
#                     (60,3,4, "lower right"),
#                     (60,5,4, "lower right"),
#                     (91,6,3, "upper right"),
#                     (118,3,3, "upper right"),
#                     (163,6,3, "upper left"),
#                     (172,2,3, "upper left")]
# for subject, trial, n_taps, loc in special_est_cases:
#     fig, ax = plot_tap_detection(subject, trial, n_taps = n_taps, legend_loc = loc)
#     fig.savefig(method_plots + "detection_{}_{}.png", dpi=300.format(subject, trial))
#     plt.close()
#
#
# ###############################
# ### Messy Data
# ###############################
#
# ## Peak estimated_locations
# for subject in range(161, 220):
#     for trial in range(1,7):
#         print(subject,trial)
#         try:
#             fig, ax = plot_tap_detection(subject, trial, n_taps = 8, legend_loc = "upper left")
#             plt.show()
#         except:
#             continue
#
# """
# 108
# 115
# 118
# 131
# 141
# 142**
# """
