## In Stage 2 of proessing, we estimate tap initiations

# Flag to create plots of output
create_plots = False

###############################
### Imports
###############################

# Warning Supression
import warnings
warnings.simplefilter("ignore")

# Standard I/O and Data Handling
import pandas as pd
import numpy as np
import os, glob
import pickle

# Signal Processing
from scripts.libraries.tap_detection import find_taps

# Data Loading
from scripts.libraries.helpers import load_tapping_data, rle, flatten
from scripts.libraries.helpers import load_pickle, dump_pickle

# Plotting
import matplotlib.pyplot as plt

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

        # Find Taps
        state_preds_detected = find_taps(signal, expected_iti, use_hmm = True, peakutils_threshold = 90)

        # Extract Inter Tap Intervals
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
