## In Stage 4 of proessing, we filter out outlier ITIs using standard deviations

###############################
### Pararameters
###############################

# Sample Rate from Sensor
sample_rate = 2000 #samples/second

# Flag to create plots
make_plots = False

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
from scripts.libraries.helpers import load_tapping_data, rle
from scripts.libraries.helpers import load_pickle, dump_pickle

# Plotting
import matplotlib.pyplot as plt

# Warning Supression
import warnings
warnings.simplefilter("ignore")

###############################
### Globals
###############################

# Main Data Directory
data_dir = "./data/"

# Tapping Data Directory
tapping_data_dir = data_dir + "tapping/"

# Tapping Filenames
tapping_filenames = glob.glob(tapping_data_dir + "*/*")
tapping_filenames = [tapping_filenames[i] for i in np.argsort([int(t.split("/")[-1].replace(".mat","")) for t in tapping_filenames])]

# Filtered Tap File (for this stage)
stage_3_processed_file = data_dir + "stage_3_processed.pickle"
stage_3_processed_taps = load_pickle(stage_3_processed_file)

# Stage 4 Tap File
stage_4_processed_file = data_dir + "stage_4_processed.pickle"
stage_4_processed_taps = {}

###############################
### Application
###############################

if not os.path.exists("./plots/stage_4/"):
    os.mkdir("./plots/stage_4/")

# Process each subject
for sub, subject_file in enumerate(tapping_filenames):

    # Parse subject identifiers
    subject_id = int(subject_file.split("/")[-1].replace(".mat",""))
    collection_date = pd.to_datetime(subject_file.split("/")[-2])

    print("Processing Subject %s" % subject_id)

    # Load Processed Taps and Raw Data
    subject_taps = stage_3_processed_taps[subject_id]
    subject_data = load_tapping_data(subject_file)

    # Split out data components
    force_signal = subject_data["trial_force"]
    preferred_period = np.ceil(subject_data["preferred_period_online"] * 2000)
    metronome_signal = subject_data["trial_metronome"]
    frequency_sequence = subject_data["frequency_sequence"]
    preferred_period_force_signal = subject_data["preferred_force"]

    # Order the taps
    if subject_taps is None:
        stage_4_processed_taps[subject_id] = dict((i, None) for i in range(1,7))
        continue
    else:
        subject_taps = [subject_taps[i] for i in range(1,7)]

    # Set up subject store
    stage_4_processed_taps[subject_id] = {}

    # Cycle through each trial
    for t, (signal, metronome, frequency, tap_inits) in enumerate(zip(force_signal, metronome_signal, frequency_sequence, subject_taps)):

        # Identify Metronome Beats and Expected ITI
        metronome = metronome/metronome.max()
        metronome_beats = np.nonzero(np.diff(abs(metronome)) == 1)[0] + 1
        last_metronome_beat = metronome_beats.max()

        # Trial frequency
        expected_iti = np.diff(metronome_beats)[0]

        # Check for discard/no taps
        if tap_inits["discard_reason"] is not None or tap_inits["tap_initiations"] is None:
            stage_4_processed_taps[subject_id][t+1] = None
            continue

        # Init separation
        trial_inits = tap_inits["tap_initiations"]
        met_inits = trial_inits[trial_inits <= 15 * 2000]
        nomet_inits = trial_inits[trial_inits > 15 * 2000]

        # ITI Formatting
        met_itis = np.array(list(zip(met_inits[:-1], met_inits[1:], np.diff(met_inits)))) # first point is the start index, second is the stop index, last is the iti
        nomet_itis = np.array(list(zip(nomet_inits[:-1], nomet_inits[1:], np.diff(nomet_inits))))

        # STD ITI cleaning (statistical outliers)
        met_itis =  met_itis[abs(met_itis[:,2] - np.median(met_itis[:,2])) <= 3*np.std(met_itis[:,2])]
        nomet_itis =  nomet_itis[abs(nomet_itis[:,2] - np.median(nomet_itis[:,2])) <= 3*np.std(nomet_itis[:,2])]

        if make_plots:

            # Create Plot
            fig, ax = plt.subplots(2,1, sharex = False, figsize = (14,6))
            ax[0].plot(signal)
            ax[0].vlines(trial_inits, ax[0].get_ylim()[0], ax[0].get_ylim()[1],
                        color = "red", linewidth = .75, linestyle = "--")
            t1 = ax[1].plot(np.arange(len(met_itis[:,2])), met_itis[:,2])
            t2 = ax[1].scatter(np.arange(len(met_itis)), met_itis[:,2], color = "red", s = 20, marker = "o")
            ax[1].plot(np.arange(len(met_itis), len(met_itis)+len(nomet_itis)), nomet_itis[:,2], color = t1[0].get_color())
            ax[1].scatter(np.arange(len(met_itis), len(met_itis)+len(nomet_itis)), nomet_itis[:,2], color = "red", s = 20, marker = "o")
            ax[1].axvline(len(met_itis) - .5, color = "black", linestyle = "--")
            if frequency != 1:
                ax[1].axhline(metronome_beats[0], color = "black", label = "Trial Frequency",
                                    alpha = 0.5, linestyle = "--")
                ax[1].axhline(int(preferred_period), color = "green", label = "Preferred",
                                    alpha = 0.5, linestyle = "--")
            else:
                ax[1].axhline(metronome_beats[0], color = "black", label = "Trial Frequency/Preferred",
                                    alpha = 0.5, linestyle = "--")
            ax[1].legend(loc = "lower right", frameon = True, facecolor = "white")
            fig.tight_layout()
            fig.suptitle("Subject %s: Trial %s" % (subject_id, t+1))
            plt.subplots_adjust(top = .945)
            fig.savefig("./plots/stage_4/%s_%s.png" % (subject_id, t+1))
            plt.close()

        # Update Store
        stage_4_processed_taps[subject_id][t+1] = {"metronome":met_itis,
                                                    "no_metronome":nomet_itis,
                                                    "inits":trial_inits,
                                                    }

# Save Results
dump_pickle(stage_4_processed_taps, stage_4_processed_file)
