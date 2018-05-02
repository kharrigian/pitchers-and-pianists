## In Stage 4 of proessing, we use monotonic span maximization to ensure we get the true tap initiation.

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
eye_checks = load_pickle(eye_check_store)
files_to_ignore = set([file for file, reason in eye_checks.items() if reason != "pass"])

# Processed Tap File (from stage 2)
stage_2_processed_file = data_dir + "stage_2_processed.pickle"
stage_2_processed_taps = load_pickle(stage_2_processed_file)

# Filtered Tap File (for this stage)
stage_3_processed_file = data_dir + "stage_3_processed.pickle"
stage_3_processed_taps = load_pickle(stage_3_processed_file)

###############################
### Monotonic Span and Initiation Detection
###############################
