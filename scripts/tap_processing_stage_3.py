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
eye_check_store = "./data/manual_inspection.pickle"

# Processed Tap File (from stage 2)
stage_2_processed_file = data_dir + "stage_2_processed.pickle"

# Filtered Tap File (for this stage)
stage_3_processed_file = data_dir + "stage_3_processed.pickle"

###############################
### Function to Load Tapping Data
###############################

# Function to Load and Transform Tapping Data
def load_tapping_data(filename):
    """
    Load .MAT file containing tapping data for a single subject.

    Args:
        filename (str): Path to the .mat file
    Returns:
        subject_data (dict): Dictionary containing subject data.
                            {
                            "preferred_force" (1d array): Signal from trial to computer preferred frequency,
                            "preferred_period_online" (float): Preferred period computed online during experiment
                            "frequency_sequence" (list of floats):Preferred period multiplier sequence
                            "trial_metronome" (2d array): Metronome signal given as auditory stimulus
                            "trial_force" (2d array): Signal from trials
                            }
    """
    subject_data = sio.matlab.loadmat(file_name = filename)["data"]
    subject_data = pd.DataFrame(subject_data[0]).transpose()
    preferred_period_force = subject_data.loc["prefForceProfile"].values[0].T[0]
    online_preferred_period_calculation = subject_data.loc['prefPeriod'][0][0][0]
    frequency_sequence = subject_data.loc["sequence"][0][0]
    trial_metronomes = np.vstack(subject_data.loc['metronome'][0][0])
    trial_force = np.vstack([i.T[0] for i in subject_data.loc['metForceProfile'][0][0]])
    return {"preferred_force":preferred_period_force, "preferred_period_online":online_preferred_period_calculation,
           "frequency_sequence":frequency_sequence, "trial_metronome":trial_metronomes, "trial_force":trial_force}

###############################
### Tap Cleaning
###############################

# Load processed taps
with open(stage_2_processed_file,"rb") as the_processed_file:
    stage_2_processed_taps = pickle.load(the_processed_file)

# Load subjects to ignore (from stage 1)
with open(eye_check_store,"rb") as the_eye_checks:
    eye_checks = pickle.load(the_eye_checks)
files_to_ignore = set([file for file, reason in eye_checks.items() if reason != "pass"])

# Initialze or Load Store for Stage 3 Processed
stage_3_processed_taps = {}
if os.path.exists(stage_3_processed_file):
    with open(stage_3_processed_file, "rb") as the_file:
        stage_3_processed_taps = pickle.load(the_file)

# Process each subject
for sub, subject_file in enumerate(tapping_filenames[:1]):

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
            met_inits = tap_inits[np.nonzero(tap_inits <= last_metronome_beat)[0]]
            nomet_inits = tap_inits[np.nonzero(tap_inits > last_metronome_beat)[0]]
            met_itis, nomet_itis = np.diff(met_inits), np.diff(nomet_inits)

            # Create Plot
            fig, ax = plt.subplots(2,1, sharex = False, figsize = (14,8))
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
            fig, ax = plt.subplots(figsize=(12,8))
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
        with open(stage_3_processed_file,"wb") as the_file:
            pickle.dump(stage_3_processed_taps, the_file, protocol = 2)

# Complete Save
print("Saving data")
with open(stage_3_processed_file,"wb") as the_file:
    pickle.dump(stage_3_processed_taps, the_file, protocol = 2)
