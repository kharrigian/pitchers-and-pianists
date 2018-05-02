## In Stage 3 of proessing, we manually remove incorrect tap identifications using an interactive GUI

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

# Data Loading
from scripts.libraries.helpers import load_tapping_data
from scripts.libraries.helpers import load_pickle, dump_pickle

# Signal Processing
import scipy.io as sio

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
eye_checks = load_pickle(eye_check_store)
files_to_ignore = set([file for file, reason in eye_checks.items() if reason != "pass"])

# Processed Tap File (from stage 2)
stage_2_processed_file = data_dir + "stage_2_processed.pickle"
stage_2_processed_taps = load_pickle(stage_2_processed_file)

# Filtered Tap File (for this stage)
stage_3_processed_file = data_dir + "stage_3_processed.pickle"
if os.path.exists(stage_3_processed_file):
    stage_3_processed_taps = load_pickle(stage_3_processed_file)
else:
    stage_3_processed_taps = {}

###############################
### Tap Cleaning
###############################

# Process each subject
for sub, subject_file in enumerate(tapping_filenames):

    # Parse subject identifiers
    subject_id = int(subject_file.split("/")[-1].replace(".mat",""))
    collection_date = pd.to_datetime(subject_file.split("/")[-2])

    print("Processing Subject %s" % subject_id)

    # Check to see if subject should be ignored
    if subject_file in files_to_ignore:
        continue

    # Check to see if subject has already been filtered
    if subject_id in stage_3_processed_taps:
        continue
    else:
        stage_3_processed_taps[subject_id] = {}

    # Load Processed Taps and Raw Data
    subject_taps = stage_2_processed_taps[subject_id]
    subject_data = load_tapping_data(subject_file)

    # Split out data components
    force_signal = subject_data["trial_force"]
    preferred_period = subject_data["preferred_period_online"]
    metronome_signal = subject_data["trial_metronome"]
    frequency_sequence = subject_data["frequency_sequence"]
    preferred_period_force_signal = subject_data["preferred_force"]

    # Order the taps
    subject_taps = [subject_taps[i] for i in range(1,7)]

    # Cycle through each trial
    for t, (signal, metronome, frequency, tap_inits) in enumerate(zip(force_signal, metronome_signal, frequency_sequence, subject_taps)):

        # Identify Metronome Beats and Expected ITI
        metronome = metronome/metronome.max()
        metronome_beats = np.nonzero(np.diff(abs(metronome)) == 1)[0] + 1
        last_metronome_beat = metronome_beats.max()
        expected_iti = np.diff(metronome_beats)[0]

        # Automatically remove initiations that occur at zero index
        tap_inits = tap_inits[np.nonzero(tap_inits>0)[0]]
        tap_inits = tap_inits[np.nonzero(tap_inits <= len(signal)-10)[0]]
        starting_tap_init_length = len(tap_inits)

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
            met_inits = tap_inits[np.nonzero(tap_inits <= last_metronome_beat + expected_iti)[0]]
            nomet_inits = tap_inits[np.nonzero(tap_inits > last_metronome_beat + expected_iti)[0]]
            met_itis, nomet_itis = np.diff(met_inits), np.diff(nomet_inits)

            # Create Plot
            fig, ax = plt.subplots(2,1, sharex = False, figsize = (14,6))
            ax[0].plot(signal)
            ax[0].vlines(tap_inits, ax[0].get_ylim()[0], ax[0].get_ylim()[1],
                        color = "red", linewidth = .75, linestyle = "--")
            t1 = ax[1].plot(np.arange(len(met_itis)), met_itis)
            t2 = ax[1].scatter(np.arange(len(met_itis)), met_itis, color = "red", s = 20, marker = "o")
            ax[1].plot(np.arange(len(met_itis), len(met_itis)+len(nomet_itis)), nomet_itis, color = t1[0].get_color())
            ax[1].scatter(np.arange(len(met_itis), len(met_itis)+len(nomet_itis)), nomet_itis, color = "red", s = 20, marker = "o")
            ax[1].axvline(len(met_itis) - .5, color = "black", linestyle = "--")
            if frequency != 1:
                ax[1].axhline(metronome_beats[0], color = "black", label = "Trial Frequency",
                                    alpha = 0.5, linestyle = "--")
                ax[1].axhline(int(preferred_period * 2000), color = "green", label = "Preferred",
                                    alpha = 0.5, linestyle = "--")
            else:
                ax[1].axhline(metronome_beats[0], color = "black", label = "Trial Frequency/Preferred",
                                    alpha = 0.5, linestyle = "--")
            ax[1].legend(loc = "lower right", frameon = True, facecolor = "white")
            fig.tight_layout()

            # Interact with Plot and Store Chosen Coordinates
            coords = []
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show(1)

            # Remove taps within selected interval
            start_len, end_len = 0, 0
            if len(coords) > 1:
                cmin, cmax = coords[0][0], coords[1][0]
                start_len = len(tap_inits)
                tap_inits = tap_inits[np.logical_not((tap_inits >= cmin ) & (tap_inits <= cmax))]
                end_len = len(tap_inits)
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
        if starting_tap_init_length != len(tap_inits):

            # Flag to note that taps were filtered
            filtered_original = True

            # Show Final Result
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(signal)
            ax.vlines(tap_inits, ax.get_ylim()[0], ax.get_ylim()[1], color = "red", linewidth = .75, linestyle = "--")
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
            tap_inits = None
        stage_3_processed_taps[subject_id][t+1] = {"tap_initiations":tap_inits,
                                                     "discard_reason":discard_reason,
                                                     "filtered":filtered_original}

    # Periodically save processed data
    if (sub + 1) % 10 == 0 :
        print("Saving data")
        dump_pickle(stage_3_processed_taps, stage_3_processed_file)

# Complete Save
print("Saving data")
dump_pickle(stage_3_processed_taps, stage_3_processed_file)
