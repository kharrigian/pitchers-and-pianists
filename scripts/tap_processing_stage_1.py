
## In Stage 1 of proessing, we manually check time series and throw out bad data (sensor malfunction, forgetting)

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

# Survey Data Filename
survey_data_filename = data_dir + "survey.csv"

###############################
### Survey Data
###############################

# Load Survey Data
survey_data = pd.read_csv(survey_data_filename)

# Add Flag re: Tapping Participation
tapping_subjects = set([int(t.split("/")[-1].split(".mat")[0]) for t in tapping_filenames])
survey_data["tapping_participant"] = survey_data.Subject.map(lambda i: i in tapping_subjects)

# Describe Arbitrary Dataset
def describe_subject_pool(survey_df):
    n_sub = len(survey_df.Subject.unique())
    female_percent = survey_df.Gender.value_counts(normalize=True)["F"] * 100
    mean_age = survey_df.Age.mean()
    std_age = survey_df.Age.std()
    return """%s subjects (%.2f%% female, %.1f+-%.1f years old) """ % (n_sub, female_percent, mean_age, std_age)

print("Entire Study: %s" % describe_subject_pool(survey_data))
print("Tapping Study: %s" % describe_subject_pool(survey_data.loc[survey_data.tapping_participant]))

###############################
### Tapping Data
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
### Bad Data Filtering (manual)
###############################

# Store Inspected File
eye_checks = {}
eye_check_store = "./data/manual_inspection.pickle"
if os.path.exists(eye_check_store):
    eye_checks = pickle.load(open(eye_check_store,"rb"))

# Check Each Subject
for file in tapping_filenames[::-1]:

    # If already inspected, continue
    if file in eye_checks:
        continue

    # Load in the Data
    subject_data = load_tapping_data(file)

    # Split out Data and Format
    subject = file.split("/")[-1].replace(".mat","")
    force_signal = subject_data["trial_force"]
    trial_data = pd.DataFrame(force_signal.T)
    trial_data.index = trial_data.index / 2000

    # Create Plot
    fig, ax = plt.subplots(figsize = (14,8), sharex = True)
    trial_data.plot(subplots = True, layout = (2,3), color = "blue", linestyle = "-", alpha = .8,
                linewidth = 1,ax = ax, legend = False)
    fig.tight_layout()
    fig.subplots_adjust(top=.94)
    fig.suptitle("Subject: %s" % subject, y = .98)
    plt.show(block=False)

    # Manually decide whether to keep or discard
    keep_or_discard = input("Discard? ")
    if len(keep_or_discard) == 0:
        eye_checks[file] = "pass"
    else:
        eye_checks[file] = keep_or_discard
    plt.close("all")

    # Save Inspection
    with open(eye_check_store, "wb") as the_file:
        pickle.dump(eye_checks, the_file, protocol = 2)

###############################
### Analyze Thrown Out Data
###############################

# Create DF
eye_check_df = pd.Series(eye_checks).reset_index().rename(columns = {"index":"file",0:"decision"})

# Absolute Thrown Out (19 out 338)
n_good_subjects = eye_check_df.decision.value_counts()["pass"]
n_bad_subjcts = len(eye_check_df) - n_good_subjects
print("%s/%s subjects thrown out immediately" % (n_bad_subjcts, len(eye_check_df)))

# Merge Demographics
eye_check_df["subject"] = eye_check_df["file"].map(lambda i: i.split("/")[-1].replace(".mat","")).astype(int)
eye_check_df = pd.merge(eye_check_df, survey_data[["Subject","Age","Gender"]], left_on = ["subject"],right_on = ["Subject"])

# Isolate Thrown Out Subjects
thrown_out_df = eye_check_df.loc[eye_check_df.decision != "pass"]

# Thrown-out Age Distribution (non-sensor related <= 15 years old)
print("Thrown Out Age Distributions")
print(thrown_out_df.groupby(["decision"]).Age.value_counts())
