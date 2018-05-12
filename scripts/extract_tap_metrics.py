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
from scripts.libraries.helpers import load_pickle, dump_pickle, load_tapping_data

# Warning Supression
import warnings
warnings.simplefilter("ignore")

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

# Load Survey Data
survey_data_filename = data_dir + "survey.csv"
survey_data = pd.read_csv(survey_data_filename).drop("Unnamed: 15",axis=1)
survey_data = survey_data.rename(columns = dict((col, col.lower()) for col in survey_data.columns))

# Tapping Filenames
tapping_filenames = glob.glob(tapping_data_dir + "*/*")
tapping_filenames = [tapping_filenames[i] for i in np.argsort([int(t.split("/")[-1].replace(".mat","")) for t in tapping_filenames])]

###############################
### Processing
###############################

# Processing Store
processed_results = []

# Process each subject
for sub, subject_file in enumerate(tapping_filenames):

    # Parse subject identifiers
    subject_id = int(subject_file.split("/")[-1].replace(".mat",""))
    collection_date = pd.to_datetime(subject_file.split("/")[-2])

    print("Processing Subject %s" % subject_id)

    # Load Processed Taps and Raw Data
    subject_taps = stage_4_processed[subject_id]
    subject_data = load_tapping_data(subject_file)

    # Move past subjects without data
    if subject_taps is None:
        continue

    # Split out data components
    preferred_period = np.ceil(subject_data["preferred_period_online"] * 2000)
    frequency_sequence = subject_data["frequency_sequence"]
    metronome_signal = subject_data["trial_metronome"]

    # Cycle through each trial
    speed_seen = []
    for trial, frequency, metronome in zip(range(1,7), frequency_sequence, metronome_signal):

        # Isolate Trial Taps
        if subject_taps[trial] is None:
            continue
        else:
            trial_tap_data = subject_taps[trial]

        # Occurence
        occurence = 1
        if frequency in speed_seen:
            occurence = 2
        else:
            speed_seen.append(frequency)

        # Identify Metronome Beats and Expected ITI
        metronome = metronome/metronome.max()
        metronome_beats = np.nonzero(np.diff(abs(metronome)) == 1)[0] + 1
        expected_iti = np.diff(metronome_beats)[0]

        # Separate Synchronization/Continuation
        met_itis = trial_tap_data['metronome']
        nomet_itis = trial_tap_data['no_metronome']

        # Important Sections
        met_mask_last5 = met_itis[:,1] >= 20000
        met_mask_last10 = met_itis[:,1] >= 10000
        nomet_mask_last15 = nomet_itis[:,1] >= metronome.shape[0] - (15*2000)
        nomet_mask_nontransient_first5 = np.logical_and(nomet_itis[:,1] >= 40000,  nomet_itis[:,1] < 50000)
        nomet_mask_last5 = nomet_itis[:,1] >= metronome.shape[0] - (5*2000)

        # Relative Synchronization Error (median - trial period) / trial_period
        met_last_5_sync_error = (np.median(met_itis[met_mask_last5][:,2]) - expected_iti) / expected_iti * 100
        met_last_10_sync_error = (np.median(met_itis[met_mask_last10][:,2]) - expected_iti) / expected_iti * 100
        nomet_last_15_sync_error = (np.median(nomet_itis[nomet_mask_last15][:,2]) - expected_iti) / expected_iti * 100
        nomet_first_5_sync_error = (np.median(nomet_itis[nomet_mask_nontransient_first5][:,2]) - expected_iti) / expected_iti * 100
        nomet_last_5_sync_error = (np.median(nomet_itis[nomet_mask_last5][:,2]) - expected_iti) / expected_iti * 100

        # Coefficient of Variation (std/median)
        met_last_5_cv = np.std(met_itis[met_mask_last5][:,2]) / np.median(met_itis[met_mask_last5][:,2])
        met_last_10_cv = np.std(met_itis[met_mask_last10][:,2]) / np.median(met_itis[met_mask_last10][:,2])
        nomet_last_15_cv = np.std(nomet_itis[nomet_mask_last15][:,2]) / np.median(nomet_itis[nomet_mask_last15][:,2])
        nomet_first_5_cv = np.std(nomet_itis[nomet_mask_nontransient_first5][:,2]) / np.median(nomet_itis[nomet_mask_nontransient_first5][:,2])
        nomet_last_5_cv = np.std(nomet_itis[nomet_mask_last5][:,2]) / np.median(nomet_itis[nomet_mask_last5][:,2])

        # Drift (pct change of first 5 median to last 5 median)
        nomet_first_5_median = np.median(nomet_itis[nomet_mask_nontransient_first5][:,2])
        nomet_last_5_median = np.median(nomet_itis[nomet_mask_last5][:,2])
        nomet_drift = (nomet_last_5_median - nomet_first_5_median) / nomet_first_5_median * 100

        # Store Metrics
        trial_metrics = {
                        ### Trial Meta ###
                        "subject":subject_id,
                        "collection_date":collection_date,
                        "trial":trial,
                        "preferred_period":preferred_period,
                        "trial_speed":frequency,
                        "speed_occurrence":occurence,
                        ### Trial Results ###
                        # Synchronization Error
                        "met_sync_error_last5":met_last_5_sync_error,
                        "met_sync_error_last10":met_last_10_sync_error,
                        "nomet_sync_error_last15":nomet_last_15_sync_error,
                        "nomet_sync_error_first5":nomet_first_5_sync_error,
                        "nomet_sync_error_last5":nomet_last_5_sync_error,
                        # Coefficient of Variation
                        "met_cv_last5":met_last_5_cv,
                        "met_cv_last10":met_last_10_cv,
                        "nomet_cv_last15":nomet_last_15_cv,
                        "nomet_cv_first5":nomet_first_5_cv,
                        "nomet_cv_last5":nomet_last_5_cv,
                        # Drift
                        "nomet_drift":nomet_drift
                        }
        processed_results.append(trial_metrics)

###############################
### Format and Merge with Survey
###############################

# Put Results into a DataFrame
processed_results = pd.DataFrame(processed_results)

# Merge with survey data
processed_results = pd.merge(processed_results, survey_data, left_on = "subject", right_on = "subject", how = "left")

# Save
processed_results.to_csv("./data/processed_results.csv", encoding = "utf-8", index = False)
