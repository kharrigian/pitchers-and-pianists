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
import statsmodels.api as sm

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

# Sample Rate
sample_rate = 2000 # samples / second

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
    preferred_period = float(subject_data["preferred_period_online"])
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
        expected_iti = np.diff(metronome_beats)[0] / sample_rate

		# Ignore Trial if Not 40 seconds long
        if len(metronome) != 40 * sample_rate:
            print("Subject %s, trial %s is not the full experiemental length" % (subject_id, trial))
            continue

        # Separate Synchronization/Continuation
        met_itis = trial_tap_data['metronome'].astype(float) / sample_rate
        nomet_itis = trial_tap_data['no_metronome'].astype(float) / sample_rate

        # Important Sections
        met_mask_last5 = np.nonzero(met_itis[:,1] >= 5.)[0]
        nomet_mask_nontransient = np.nonzero(nomet_itis[:, 1] >= 20.)[0]

        # If either masks don't have minimum number of taps, break
        if len(met_mask_last5) < 5 or len(nomet_mask_nontransient) < 10:
            continue

        # Isolate ITIs within each Sections
        met_itis_last5 = met_itis[met_mask_last5][-5:, 2]
        nomet_itis_first5 = nomet_itis[nomet_mask_nontransient][:5, 2]
        nomet_itis_last5 = nomet_itis[nomet_mask_nontransient][-5:, 2]

        # Absolute Synchronization Error (median - trial_period)
        met_last_5_sync_error = np.median(met_itis_last5) - expected_iti
        nomet_first_5_sync_error = np.median(nomet_itis_first5) - expected_iti
        nomet_last_5_sync_error = np.median(nomet_itis_last5) - expected_iti

        # Relative Synchronization Error (median - trial period) / trial_period
        met_last_5_sync_error_rel = (max(met_last_5_sync_error, 1./sample_rate) if met_last_5_sync_error >= 0 else met_last_5_sync_error) / expected_iti * 100
        nomet_first_5_sync_error_rel = (max(nomet_first_5_sync_error, 1./sample_rate) if nomet_first_5_sync_error >= 0 else nomet_first_5_sync_error) / expected_iti * 100
        nomet_last_5_sync_error_rel = (max(nomet_last_5_sync_error, 1./sample_rate) if nomet_last_5_sync_error >= 0 else nomet_last_5_sync_error) / expected_iti * 100

        # Coefficient of Variation (std/mean)
        met_last_5_cv = np.std(met_itis_last5) / np.mean(met_itis_last5)
        nomet_first_5_cv = np.std(nomet_itis_first5) / np.mean(nomet_itis_first5)
        nomet_last_5_cv = np.std(nomet_itis_last5) / np.mean(nomet_itis_last5)

        # Drift (pct change of first 5 median to last 5 median)
        met_last_5_median = np.median(met_itis_last5)
        nomet_first_5_median = np.median(nomet_itis_first5)
        nomet_last_5_median = np.median(nomet_itis_last5)
        # nomet_drift = nomet_last_5_median - nomet_first_5_median
        nomet_drift = nomet_last_5_median - met_last_5_median
        # nomet_drift_rel = (max(nomet_drift, 1./sample_rate) if nomet_drift >= 0 else nomet_drift) / nomet_first_5_median * 100
        nomet_drift_rel = (max(nomet_drift, 1./sample_rate) if nomet_drift >= 0 else nomet_drift) / met_last_5_median * 100

		# Drift (Regression Coefficient)
        y = nomet_itis[nomet_mask_nontransient][:,2]
        x = np.arange(len(y)); x = x/x.max(); x = sm.add_constant(x)
        model = sm.OLS(y, x).fit()
        nomet_drift_regression = model.params[1]

        # Store Metrics
        trial_metrics = {
                        ### Trial Meta ###
                        "subject":subject_id,
                        "collection_date":collection_date,
                        "trial":trial,
                        "preferred_period":preferred_period * 1000,
                        "trial_speed":frequency,
                        "speed_occurrence":occurence,
                        ### Trial Results ###
                        # Synchronization Error
                        "met_sync_error_last5":met_last_5_sync_error * 1000,
                        "nomet_sync_error_first5":nomet_first_5_sync_error * 1000,
                        "nomet_sync_error_last5":nomet_last_5_sync_error * 1000,
                        # Relative Synchronization Error
                        "met_sync_error_last5_rel":met_last_5_sync_error_rel,
                        "nomet_sync_error_first5_rel":nomet_first_5_sync_error_rel,
                        "nomet_sync_error_last5_rel":nomet_last_5_sync_error_rel,
                        # Coefficient of Variation
                        "met_cv_last5":met_last_5_cv,
                        "nomet_cv_first5":nomet_first_5_cv,
                        "nomet_cv_last5":nomet_last_5_cv,
                        # Drift
                        "nomet_drift":nomet_drift * 1000,
                        "nomet_drift_rel":nomet_drift_rel,
                        "nomet_drift_regression":nomet_drift_regression
                        }
        processed_results.append(trial_metrics)

###############################
### Format and Merge with Survey
###############################

# Put Results into a DataFrame
processed_results = pd.DataFrame(processed_results)

# Merge with survey data
processed_results = pd.merge(processed_results, survey_data, left_on = "subject", right_on = "subject", how = "left")

# Replace Survey Blanks
processed_results.replace("BLANK", np.nan, inplace = True)

# Standardize Column Names
processed_results = processed_results.rename(columns = dict((col, col.replace(" ","_")) for col in processed_results.columns))

# Encode Gender
processed_results["gender"] = processed_results["gender"].map(lambda val: {"M":0,"F":1}[val] if not pd.isnull(val) else None)

# Encode Musical Experience
processed_results["musical_experience"] = processed_results["musical_experience"].map(lambda val: {"N":0,"Y":1}[val] if not pd.isnull(val) else None)
processed_results["musical_experience_yrs"] = processed_results["specify_musical_experience"].map(lambda i: max([-1] + [float(j) for j in i.split() if j.isdigit()]) if not pd.isnull(i) else 0)

# Encode Athletic Experience
processed_results["sport_experience"] = processed_results["sport_experience"].map(lambda val: {"N":0,"Y":1}[val] if not pd.isnull(val) else None)
processed_results["sport_experience_yrs"] = processed_results["specify_sports_experience"].map(lambda i: max([-1] + [float(j) for j in i.split() if j.isdigit()]) if not pd.isnull(i) else 0)

# Encode Disorder
processed_results["disorders"] = processed_results["disorders"].map(lambda val: {"N":0,"Y":1}[val] if not pd.isnull(val) else None)

# Encode Trial Speed
processed_results["trial_speed"] = processed_results["trial_speed"].map(lambda val: {0.8:"SpedUp", 1.0:"NoChange", 1.2:"SlowedDown"}[val] )

# Save
processed_results.to_csv("./data/processed_results.csv", encoding = "utf-8", index = False)
