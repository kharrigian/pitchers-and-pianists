
## In Stage 1 of proessing, we manually check time series and throw out bad data (sensor malfunction, forgetting the task)

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

# Data Loading
from scripts.libraries.helpers import load_tapping_data
from scripts.libraries.helpers import load_pickle, dump_pickle

# Plotting
import matplotlib.pyplot as plt

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
tapping_subjects = [int(t.split("/")[-1].split(".mat")[0]) for t in tapping_filenames]

# Survey Data Filename
survey_data_filename = data_dir + "survey.csv"

# Store Inspected File
eye_checks = {}
eye_check_store = data_dir + "manual_inspection.pickle"
if os.path.exists(eye_check_store):
    eye_checks = load_pickle(eye_check_store)

###############################
### Survey Data
###############################

# Load Survey Data
survey_data = pd.read_csv(survey_data_filename)

# Missing Survey Subjects
missing_survey_data = [t for t in tapping_subjects if int(t) not in survey_data.Subject.values]

# Add Flag re: Tapping Participation
survey_data["valid_tapping_participant"] = survey_data.Subject.map(lambda i: i in tapping_subjects)

# Describe Arbitrary Dataset
def describe_subject_pool(survey_df):
    n_sub = len(survey_df.Subject.unique())
    female_percent = survey_df.Gender.value_counts(normalize=True)["F"] * 100
    mean_age = survey_df.Age.mean()
    std_age = survey_df.Age.std()
    return """%s subjects (%.2f%% female, %.1f+-%.1f years old) """ % (n_sub, female_percent, mean_age, std_age)

print("Entire Study: %s" % describe_subject_pool(survey_data))
print("Tapping Study: %s" % describe_subject_pool(survey_data.loc[survey_data.valid_tapping_participant]))

###############################
### Bad Data Filtering (manual)
###############################

# Stage 1 (Bad Trial Plots)
stage_1_dir = "./plots/stage_1/"
if not os.path.exists(stage_1_dir):
    os.mkdir(stage_1_dir)

# Check Each Subject
for file in tapping_filenames:

    # Load in the Data
    subject_data = load_tapping_data(file)

    # Split out Data and Format
    subject = file.split("/")[-1].replace(".mat","")
    force_signal = subject_data["trial_force"]
    trial_data = pd.DataFrame(force_signal.T)
    trial_data.index = trial_data.index / 2000

    # If subject didn't fill out survey, ignore
    if int(subject) in missing_survey_data:
        eye_checks[file] = "missing_survey_data"

    # Check subjects that haven't been inspected or were discarded
    if file not in eye_checks or (file in eye_checks and eye_checks[file] != "pass"):

        # Create Plot
        fig, ax = plt.subplots(figsize = (14,8), sharex = True)
        trial_data.plot(subplots = True, layout = (2,3), color = "blue", linestyle = "-", alpha = .8,
                    linewidth = 1, ax = ax, legend = False)
        fig.tight_layout()
        fig.subplots_adjust(top=.94)
        fig.suptitle("Subject: %s" % subject, y = .98)

        # If already inspected, continue
        if file not in eye_checks:

            plt.show(block=False)

            # Manually decide whether to keep or discard
            keep_or_discard = input("Discard? ")
            if len(keep_or_discard) == 0:
                keep_or_discard == "pass"

        # Otherwise, load decision
        else:
            keep_or_discard = eye_checks[file]

        # Store Decision
        eye_checks[file] = keep_or_discard

        # Save plot if discarded
        if keep_or_discard != "pass":
            plt.savefig(stage_1_dir + "%s_%s.png" % (subject, keep_or_discard))

        # Close Plot
        plt.close("all")

    # Save Inspection
    dump_pickle(eye_checks, eye_check_store)

###############################
### Analyze Thrown Out Data
###############################

# Create DF
eye_check_df = pd.Series(eye_checks).reset_index().rename(columns = {"index":"file",0:"decision"})

# Absolute Thrown Out (18 out 338)
n_good_subjects = eye_check_df.decision.value_counts()["pass"]
n_bad_subjcts = len(eye_check_df) - n_good_subjects
print("%s/%s subjects thrown out immediately" % (n_bad_subjcts, len(eye_check_df)))

# Merge Demographics
eye_check_df["subject"] = eye_check_df["file"].map(lambda i: i.split("/")[-1].replace(".mat","")).astype(int)
eye_check_df = pd.merge(eye_check_df, survey_data[["Subject","Age","Gender"]], left_on = ["subject"], right_on = ["Subject"], how = "left")

# Isolate Thrown Out Subjects
thrown_out_df = eye_check_df.loc[eye_check_df.decision != "pass"]

# Thrown-out Age Distribution (non-sensor related <= 15 years old)
print("Thrown Out Age Distributions")
print(thrown_out_df.groupby(["decision"]).Age.value_counts())

# Valid Subject Pool
print(describe_subject_pool(eye_check_df.loc[eye_check_df.decision == "pass"]))

###############################
### Notes
###############################

"""

We begin data analysis with 336 subject data files. However, we immediately throw out
1 subject who did not complete a survey at all (subject 105).

Then we throw out an additional 17 subjects for the following reasons:
* forget (3): the subject forgot to continue tapping after the metronome ended in more than 2 trials
* style (2): the subject tapped too lightly or pressed in an abnormal fashion on the sensor
* sensor (12): the subjects data was corrupted by a sensor malfunction and we do not expect to recover taps correctly

This leaves us with 318 subjects for which to process (identify taps, run analysis, etc.). Their summary is as follow:
318 subjects (54.09% female, 26.0+-14.4 years old)
"""
