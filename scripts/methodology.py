
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
import statsmodels.api as sm

# Data Loading
from scripts.libraries.helpers import load_pickle, dump_pickle, load_tapping_data

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
### Plotting Functions
###############################

def plot_trial_methods(subject, trial_number):
    """
    Plot results from a subject and trial
    """
    if subject not in tapping_data:
        raise ValueError("Subject {} does not have any usable data".format(subject))
    ## Identify relevant data
    subject_data = tapping_data[subject_id]
    preferred_period = float(subject_data["preferred_period_online"]) * 1000
    trial_frequency = subject_data["frequency_sequence"][trial_number-1]
    metronome_signal = subject_data["trial_metronome"][trial_number-1]
    trial_force = subject_data["trial_force"][trial_number-1]
    trial_taps = subject_data["processed_taps"][trial_number]
