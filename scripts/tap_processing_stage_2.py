## In Stage 2 of proessing, we estimate tap initiations

# Flag to create plots of output
create_plots = False

###############################
### Imports
###############################

# Standard I/O and Data Handling
import pandas as pd
import numpy as np
import os, glob, sys
import datetime
import copy
import pickle

# Signal Processing
import scipy.io as sio
import statsmodels.api as sm
from sklearn import preprocessing

# Data Loading
from scripts.libraries.helpers import load_tapping_data, rle, flatten
from scripts.libraries.helpers import load_pickle, dump_pickle

# Model
from hmmlearn import hmm

# Plotting
import matplotlib.pyplot as plt

# Warning Supression
import warnings
warnings.simplefilter("ignore")

###############################
### Globals
###############################

# Sample Rate from Sensor
sample_rate = 2000 #samples/second

# Main Data Directory
data_dir = "./data/"

# Tapping Data Directory
tapping_data_dir = data_dir + "tapping/"

# Tapping Filenames
tapping_filenames = glob.glob(tapping_data_dir + "*/*")
tapping_filenames = [tapping_filenames[i] for i in np.argsort([int(t.split("/")[-1].replace(".mat","")) for t in tapping_filenames])]

# Manual Inspections (to ignore)
eye_check_store = data_dir + "manual_inspection.pickle"
eye_checks = pickle.load(open(eye_check_store,"rb"))
files_to_ignore = set([file for file, reason in eye_checks.items() if reason != "pass"])

# Initialize or Load Store for Processed Data
stage_2_processed_tap_file = "./data/stage_2_processed.pickle"
stage_2_processed = {}
if os.path.exists(stage_2_processed_tap_file):
    stage_2_processed = load_pickle(stage_2_processed_tap_file)

# Stage 2 Plot Directory
plot_dir = "./plots/stage_2/"
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

###############################
### Processing Functions
###############################

# Normalization of signal between 0 and 1 (within windows)
def min_max_normalization(signal):
    return preprocessing.MinMaxScaler((0,1)).fit_transform(signal.reshape(-1,1)).T[0]

# Parameterize base HMM
base_model = hmm.GaussianHMM(n_components=2, covariance_type='diag', min_covar=0.0001,
                         transmat_prior=1.0, means_prior=0, means_weight=0, covars_prior=0.03,
                         covars_weight=1.5, algorithm='viterbi', random_state=42,
                         n_iter=100, tol=0.001, verbose=False, params='stmc', init_params='stmc')

# Tap Initiation
def estimate_tap_initiation(signal, estimated_locations, expected_iti):
    """
    Zero in on the true tap initiations
    """
    window_edges = [int(max([estimated_locations[0] - .2*expected_iti,0]))] + \
                    [int(np.mean(estimated_locations[i:i+2])) for i in range(len(estimated_locations)-1)] + \
                    [int(min([estimated_locations[-1] + .2*expected_iti,len(signal)-1]))]
    filtered_inits = []
    for start, stop, est in zip(window_edges[:-1],window_edges[1:], estimated_locations):
        signal_window = signal[start:stop]
        signal_window_normalized = min_max_normalization(signal_window)
        # clean the signal
        decomposed = sm.tsa.seasonal_decompose(signal_window_normalized, model = "additive", freq = 10)
        signal_denoised = pd.Series(decomposed.trend).fillna(method = "ffill").fillna(method="bfill").values
        # remodel around nice window
        mod = copy.copy(base_model)
        try:
            mod.fit(signal_denoised.reshape(-1,1))
            pred = mod.predict(signal_denoised.reshape(-1,1))
        except:
            continue
        active_state = pred[np.argmax(signal_denoised)]
        pred = list(map(lambda p: 1 if p == active_state else 0, pred))
        # extract state changes
        statelens, statepos, states = rle(pred)
        state_preds_detected, statelens_detected = statepos[np.nonzero(states)[0]], statelens[np.nonzero(states)[0]]
        # Require Monotonic Increasing/Constant
        increasing = pd.Series(signal_denoised).diff().fillna(0) > 0
        state_preds_detected = [s for s in state_preds_detected if increasing[s-5:s+6].sum() > 3]
        # Append Guesses
        if len(state_preds_detected) > 0:
            # Might be multiple
            estimates = [min(state_preds_detected)]
            last_estimate = estimates[0]
            for s in state_preds_detected[1:]:
                if s - last_estimate > (expected_iti/3):
                    estimates.append(s)
                    last_estimate = s
            # Append all estimates
            for e in estimates:
                filtered_inits.append(start + e)
    return np.array(filtered_inits)


###############################
### Tap Identification
###############################


# Process each subject
for sub, subject_file in enumerate(tapping_filenames):

    # Parse subject identifiers
    subject_id = int(subject_file.split("/")[-1].replace(".mat",""))
    collection_date = pd.to_datetime(subject_file.split("/")[-2])

    print("Processing Subject %s" % subject_id)

    # Check to see if subject has already been processed
    if subject_id in stage_2_processed:
        continue

    # Create store for processed subject data
    stage_2_processed[subject_id] = {}

    # Check to see if subject should be ignored
    if subject_file in files_to_ignore:
        stage_2_processed[subject_id] = None
        continue

    # Load Subject Data
    subject_data = load_tapping_data(subject_file)

    # Split out data components
    force_signal = subject_data["trial_force"]
    preferred_period = subject_data["preferred_period_online"]
    metronome_signal = subject_data["trial_metronome"]
    frequency_sequence = subject_data["frequency_sequence"]
    preferred_period_force_signal = subject_data["preferred_force"]

    # Cycle through each trial
    for t, (signal, metronome, frequency) in enumerate(zip(force_signal, metronome_signal, frequency_sequence)):

        # Identify Metronome Beats and Expected ITI
        metronome = metronome/metronome.max()
        metronome_beats = np.nonzero(np.diff(abs(metronome)) == 1)[0] + 1
        expected_iti = np.diff(metronome_beats)[0]

        # Extract Main Trend in Signal
        decomposition = sm.tsa.seasonal_decompose(signal, model = "additive", freq = 50)
        signal_denoised = pd.Series(decomposition.trend).fillna(method = "ffill").fillna(method="bfill").values

        # Quantize Signal (base = rounding integer)
        quantizer = lambda x, base: int(base * round(float(x)/base))
        signal_quantized = np.array([quantizer(f, 2) for f in (min_max_normalization(signal_denoised) * 100)])

        # Create Differently Sized Windows
        windows_large_range = list(np.arange(0, len(signal_quantized), expected_iti * 5)) + [len(signal_quantized)]
        windows_small_range = list(np.arange(0, len(signal_quantized), expected_iti * 2)) + [len(signal_quantized)]
        windows_global_range = [0, len(signal_quantized)]

        # Fit and Predict HMM within each window
        all_preds = []
        for windows in [windows_small_range, windows_large_range, windows_global_range]:
            state_preds = []
            for w_start, w_end in zip(windows[:-1],windows[1:]):
                window_sig = signal_quantized[w_start:w_end]
                window_sig = np.floor(min_max_normalization(window_sig) * 100)
                window_hmm = copy.copy(base_model)
                window_hmm.fit(window_sig.reshape(-1,1))
                try:
                    wpred = window_hmm.predict(window_sig.reshape(-1,1))
                    active_state = wpred[np.argmax(window_sig)]
                    wpred = list(map(lambda i: 1 if i == active_state else 0, wpred))
                except:
                    wpred = [0 for i in window_sig]
                state_preds.append(wpred)
            state_preds = np.array(flatten(state_preds))
            all_preds.append(state_preds)

        # Require All Scales to Predict and Active State
        state_preds = (np.array(all_preds).T.sum(axis = 1) == 3).astype(int)

        # Impact Detection using Estimated States
        statelens, statepos, states = rle(state_preds)
        state_preds_detected = statepos[np.nonzero(states)[0]]

        # Require Monotonic Increasing/Constant
        increasing = pd.Series(signal_quantized).diff().fillna(0) > 0
        state_preds_detected = [s for s in state_preds_detected if increasing[s-5:s+6].sum() > 0]

        # Filter out double taps
        min_delay = expected_iti / 3
        last_tap = 0
        filtered_preds_detected = []
        for pred in state_preds_detected:
            if pred - last_tap >= min_delay:
                filtered_preds_detected.append(pred)
                last_tap = pred
        state_preds_detected = np.array(filtered_preds_detected)

        # Get as close to tap initiation as possible
        state_preds_detected = estimate_tap_initiation(signal_denoised, state_preds_detected, expected_iti)

        # Inter Tap Intervals
        met_ends = np.nonzero(state_preds_detected > metronome_beats.max() + expected_iti)[0].min()
        met_itis, nomet_itis = np.diff(state_preds_detected[:met_ends+1]), np.diff(state_preds_detected[met_ends+1:])

        # Plot Results if desired
        if create_plots:

            fig, axes = plt.subplots(2,1, figsize = (14,6), sharex = False)
            axes[0].plot(signal, alpha = 0.7, linewidth = 1)
            axes[0].vlines(state_preds_detected, axes[0].get_ylim()[0], axes[0].get_ylim()[1], color = "red", linewidth = .75, linestyle = "--")
            t1 = axes[1].plot(np.arange(len(met_itis)), met_itis)
            t2 = axes[1].scatter(np.arange(len(met_itis)), met_itis, color = "red", s = 20, marker = "o")
            axes[1].plot(np.arange(met_ends, met_ends+len(nomet_itis)), nomet_itis, color = t1[0].get_color())
            axes[1].scatter(np.arange(met_ends, met_ends+len(nomet_itis)), nomet_itis, color = "red", s = 20, marker = "o")
            axes[1].axvline(met_ends - .5, color = "black", linestyle = "--")
            if frequency != 1:
                axes[1].axhline(metronome_beats[0], color = "black", label = "Trial Frequency",
                                    alpha = 0.5, linestyle = "--")
                axes[1].axhline(int(preferred_period * 2000), color = "green", label = "Preferred",
                                    alpha = 0.5, linestyle = "--")
            else:
                axes[1].axhline(metronome_beats[0], color = "black", label = "Trial Frequency/Preferred",
                                    alpha = 0.5, linestyle = "--")
            axes[1].legend(loc = "lower right", frameon = True, facecolor = "white")
            axes[0].set_xlim(-.5, len(signal)+.5)
            axes[1].set_xlim(-.5, len(state_preds_detected)-1)
            fig.tight_layout()
            fig.suptitle("Subject %s: Trial %s" % (subject_id, t+1), y = .98)
            fig.subplots_adjust(top = .94)
            fig.savefig(plot_dir + "%s_%s.png" % (subject_id, t+1))
            plt.close("all")

        # Save Tap Initiations
        stage_2_processed[subject_id][t+1] = state_preds_detected

    # Periodically save processed data
    if (sub + 1) % 20 == 0 :
        print("Saving data")
        dump_pickle(stage_2_processed, stage_2_processed_tap_file)

# Complete Save
print("Saving data")
with open(stage_2_processed_tap_file,"wb") as the_file:
    dump_pickle(stage_2_processed, stage_2_processed_tap_file)
