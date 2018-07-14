
### Visualizations to showcase the methodology

##############################
### Imports
###############################

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
from hmmlearn import hmm
import peakutils
from sklearn import preprocessing

# Plotting
import matplotlib.pyplot as plt

# Warning Supression
import warnings
warnings.simplefilter("ignore")

###############################
###  Plot Helpers
###############################

## Plotting Variables
standard_fig = (10,5.8)
plot_dir = "./plots/"
method_plots = plot_dir + "methodology/"

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


###############################
### Detection Helpers
###############################

# Normalization of signal between 0 and 1 (within windows)
def min_max_normalization(signal):
    return preprocessing.MinMaxScaler((0,1)).fit_transform(signal.reshape(-1,1)).T[0]

# Parameterize base HMM
base_model = hmm.GaussianHMM(n_components=2, covariance_type='diag', min_covar=0.0001,
                         transmat_prior=1.0, means_prior=0, means_weight=0, covars_prior=0.03,
                         covars_weight=1.5, algorithm='viterbi', random_state=42,
                         n_iter=100, tol=0.001, verbose=False, params='stmc', init_params='stmc')

def demo_infer_tap_states(signal, min_dist = 1):
    """
    Stripped down version of the HMM-based tap identification process for demo purposes
    """
    signal_normalized = min_max_normalization(signal)
    ## HMM Tap Identification
    signal_hmm = copy.copy(base_model)
    signal_hmm.fit(signal_normalized.reshape(-1,1))
    try:
        wpred = signal_hmm.predict(signal_normalized.reshape(-1,1))
        active_state = wpred[np.argmax(signal_normalized)]
        wpred = list(map(lambda i: 1 if i == active_state else 0, wpred))
    except:
        wpred = [0 for i in signal_normalized]
    ## Peakutils Tap Identification
    peak_indexes = peakutils.indexes(signal_normalized, thres = .2, min_dist = min_dist)
    return signal_normalized, np.array(wpred), peak_indexes

###############################
### Plotting Functions
###############################

def plot_trial_methods(subject, trial_number):
    """
    Plot results from a subject and trial
    """
    if subject not in tapping_data:
        raise ValueError("Subject {} does not have any usable data".format(subject))
    ## Identify relevant data
    subject_data = tapping_data[subject]
    preferred_period = float(subject_data["preferred_period_online"]) * 1000
    trial_frequency = subject_data["frequency_sequence"][trial_number-1]
    metronome_signal = subject_data["trial_metronome"][trial_number-1]
    trial_force = subject_data["trial_force"][trial_number-1]
    trial_taps = subject_data["processed_taps"][trial_number]
    ## Plot #1 -- Experiment Design
    fig, ax = plt.subplots(2, 1, figsize = standard_fig, sharex = False, sharey = False)
    ax[0].fill_between([5,15], min(trial_force), max(trial_force)*1.1, color = "lightgray", alpha = .2)
    ax[0].fill_between([20,40], min(trial_force), max(trial_force)*1.1, color = "lightgray", alpha = .2, label = "Non-Transient")
    ax[0].plot(np.arange(len(trial_force))/2000., trial_force, color = "black", linewidth = 2)
    ax[0].axvline(15., linestyle = "--", linewidth = 2, color = "red")
    ax[0].set_xlabel("Time (s)", fontsize = 16, labelpad = 5); ax[0].set_ylabel("Force", fontsize = 16, labelpad = 5)
    ax[0].set_xlim(0,40.)
    ax[0].set_ylim(min(trial_force), max(trial_force) * 1.1)
    met_ind = np.arange(len(trial_taps["metronome"]))+1
    nomet_ind = np.arange(max(met_ind)+1, max(met_ind)+1 + len(trial_taps["no_metronome"]))
    ax[1].plot(met_ind, trial_taps["metronome"][:,2]/2., color = "black", linewidth = 2, linestyle = "-")
    ax[1].scatter(met_ind, trial_taps["metronome"][:,2]/2., s = 50, color = "lightgray", edgecolor = "black", zorder = 3)
    ax[1].plot(nomet_ind, trial_taps["no_metronome"][:,2]/2., color = "black", linewidth = 2, linestyle = "-")
    ax[1].scatter(nomet_ind, trial_taps["no_metronome"][:,2]/2., s = 50, color = "lightgray", edgecolor = "black", zorder = 3)
    ax[1].axvline(max(met_ind) + .5, linewidth = 2, linestyle = "--", color = "red", label = "Metronome Ends")
    fb_min = min(min(trial_taps["metronome"][:,2]/2),min(trial_taps["no_metronome"][:,2]/2)) * .9
    fb_max = max(max(trial_taps["metronome"][:,2]/2),max(trial_taps["no_metronome"][:,2]/2)) * 1.1
    ax[1].fill_between([np.nonzero(trial_taps["metronome"][:,0] >= 10000)[0][0]+1,max(met_ind)],
                        fb_min, fb_max, color = "lightgray", alpha = .2)
    ax[1].fill_between([min(nomet_ind) + min(np.nonzero(trial_taps["no_metronome"][:,0] >= 40000.)[0]), max(nomet_ind)],
                        fb_min, fb_max, color = "lightgray", alpha = .2, label = "Non-Transient")
    ax[1].axhline(preferred_period, linestyle = "--", color = "black", linewidth = 1.5)
    ax[1].text(max(nomet_ind)+1, preferred_period, "Preferred/Trial\nPeriod" if trial_frequency == 1 else "Preferred Period",
                    fontsize = 14, va = "center", multialignment = "center")
    if trial_frequency != 1:
        ax[1].axhline(preferred_period * trial_frequency, linestyle = "--", color = "black", linewidth = 1.5)
        ax[1].text(max(nomet_ind) + 1, preferred_period * trial_frequency, "Trial Period",  fontsize = 14, va = "center")
    ax[1].set_xlabel("Tap #", fontsize = 16, labelpad = 5)
    ax[1].set_ylabel("Inter Tap\nInterval (ms)", fontsize = 16, labelpad = 5, multialignment = "center")
    ax[1].set_xlim(-.5, max(nomet_ind)+.5)
    for a in ax:
        a.tick_params(labelsize = 14)
    ax[1].set_ylim(fb_min, fb_max)
    handles, labels = ax[1].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc = (.25, .935), ncol = 2, fontsize = 14, borderpad = .2, handlelength = 2)
    for t in leg.texts:  t.set_multialignment('center')
    fig.tight_layout()
    fig.subplots_adjust(right = .825, top = .925)
    return fig, ax

def plot_tap_detection(subject, trial_number, n_taps = 6, lpad = 50, rpad = 400, legend_loc = "upper right"):
    if subject not in tapping_data:
        raise ValueError("Subject {} does not have any usable data".format(subject))
    ## Identify relevant data
    subject_data = tapping_data[subject]
    preferred_period = float(subject_data["preferred_period_online"]) * 1000
    trial_frequency = subject_data["frequency_sequence"][trial_number-1]
    metronome_signal = subject_data["trial_metronome"][trial_number-1]
    trial_force = subject_data["trial_force"][trial_number-1]
    trial_taps = subject_data["processed_taps"][trial_number]
    ## Plot #2 -- Tap Detection
    signal_subset = trial_taps["metronome"][4:4+n_taps,0]
    force_signal_subset = trial_force[signal_subset.min()-lpad: signal_subset.max()+rpad]
    sig_normed, hmm_states, sig_peaks = demo_infer_tap_states(force_signal_subset,
                                        min_dist = int(preferred_period  * trial_frequency * .6))
    subset_time = np.arange(signal_subset.min()-lpad,signal_subset.max()+rpad)/2000
    fig, ax = plt.subplots(1, 1, figsize = standard_fig)
    ax.plot(subset_time, sig_normed, color = "black", linewidth = 2, label = "Signal")
    ax.plot(subset_time, hmm_states, color = "red", linestyle = "--", linewidth = 1.5, label = "HMM State", alpha = .8)
    ax.scatter(subset_time[sig_peaks], sig_normed[sig_peaks], s = 100, color = "lightgray", edgecolor = "black", zorder = 3,
                label = "Standard Peak Detection")
    ax.set_xlim(min(subset_time), max(subset_time))
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Time (s)", fontsize = 16, labelpad = 5)
    ax.set_ylabel("Force", fontsize = 16, labelpad = 5)
    ax.tick_params(labelsize = 14)
    ax.legend(loc = legend_loc, frameon = True, facecolor = "white", framealpha = 1, fontsize = 14)
    fig.tight_layout()
    return fig, ax

###############################
### Example
###############################

## Trial Methods
special_methods_cases = [(61,2),(69,3),(80,2),(161,2),(179,3),(194,3),(250,3),
                            (258,1),(268,4),(287,5),(307,4),(310,3),(314,3),(314,6)]
for subject, trial in special_methods_cases:
    fig, ax = plot_trial_methods(subject, trial)
    fig.savefig(method_plots + "{}_{}.png".format(subject, trial))
    plt.close()

## Peak estimated_locations
special_est_cases = [(55,4,4, "lower right"),
                    (60,3,4, "lower right"),
                    (60,5,4, "lower right"),
                    (91,6,3, "upper right"),
                    (118,3,3, "upper right"),
                    (163,6,3, "upper left"),
                    (172,2,3, "upper left")]
for subject, trial, n_taps, loc in special_est_cases:
    fig, ax = plot_tap_detection(subject, trial, n_taps = n_taps, legend_loc = loc)
    fig.savefig(method_plots + "detection_{}_{}.png".format(subject, trial))
    plt.close()
