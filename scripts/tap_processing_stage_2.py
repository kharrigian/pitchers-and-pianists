## In Stage 2 of proessing, we estimate tap initiations

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
import statsmodels.api as sm
from sklearn import preprocessing

# Model
from hmmlearn import hmm

# Plotting
import matplotlib.pyplot as plt

# Warning Supression
import warnings
warnings.simplefilter("ignore")

###############################
### Helpers
###############################

flatten = lambda l: [item for sublist in l for item in sublist]

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

# Survey Data Filename
survey_data_filename = data_dir + "survey.csv"

# Manual Inspections (to ignore)
eye_check_store = "./data/manual_inspection.pickle"

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
### Processing Functions
###############################

# Normalization of signal between 0 and 1 (within windows)
def max_window_normalization(signal, window_size):
    windows = list(np.arange(0, len(signal), window_size)) + [len(signal)]
    normalized_signal = []
    for w_start, w_end in zip(windows[:-1],windows[1:]):
        window_sig = signal[w_start:w_end]
        normalized_signal.append(preprocessing.MinMaxScaler((0,1)).fit_transform(window_sig.reshape(-1,1)).T[0])
    return np.array(flatten(normalized_signal))

# Run Length
def rle(inarray):
    """ run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values) """
    ia = np.asarray(inarray)                  # force numpy
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = np.array(ia[1:] != ia[:-1])     # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)   # must include last element posi
        z = np.diff(np.append(-1, i))       # run lengths
        p = np.cumsum(np.append(0, z))[:-1] # positions
        return(z, p, ia[i])

# Function to use estimated initiation and look for maximum accelaration as better estimate
def find_tap_initiation(force_signal, estimated_location, window_start=None, window_end=None):
    if window_start is None:
        window_start = estimated_location - 100
    if window_end is None:
        window_end = estimated_location + 100
    force_window = force_signal[window_start:window_end]
    second_force_diff = (pd.Series(force_window).shift(-2) - pd.Series(force_window).shift(2))
    second_force_diff = second_force_diff.fillna(method = "ffill").fillna(method = "bfill")
    for std_cutoff in [3,2,1]:
        is_spike = second_force_diff - np.median(second_force_diff) >= std_cutoff*np.std(second_force_diff)
        is_spike_shift = np.roll(is_spike, 1)
        starts = is_spike & ~is_spike_shift
        if len(np.nonzero(starts.values)[0]) > 0:
            spike_ind = np.min(np.nonzero(starts.values)[0])
            return window_start + spike_ind
    return estimated_location

###############################
### Tap Identification
###############################

# Load subjects to ignore
eye_checks = pickle.load(open(eye_check_store,"rb"))
files_to_ignore = set([file for file, reason in eye_checks.items() if reason != "pass"])

# Initialize or Load Store for Processed Data
stage_2_processed_tap_file = "./data/stage_2_processed.pickle"
stage_2_processed = {}
if os.path.exists(stage_2_processed_tap_file):
    stage_2_processed = pickle.load(open(stage_2_processed_tap_file, "rb"))

# Plot Directory
plot_dir = "./plots/stage_2/"
if not os.path.exists(plot_dir):
    os.mkdir(plot_dir)

# Process each subject
for sub, subject_file in enumerate(tapping_filenames):

    # Parse subject identifiers
    subject_id = int(subject_file.split("/")[-1].replace(".mat",""))
    collection_date = pd.to_datetime(subject_file.split("/")[-2])

    print("Processing Subject %s" % subject_id)

    # Check to see if subject should be ignored
    if subject_file in files_to_ignore:
        continue

    # Check to see if subject has already been processed
    if subject_id in stage_2_processed:
        continue

    # Create store for processed subject data
    stage_2_processed[subject_id] = {}

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

        # Trend in Signal
        decomposition = sm.tsa.seasonal_decompose(signal, model = "additive", freq = 100)
        signal_denoised = pd.Series(decomposition.trend).fillna(method = "ffill").fillna(method="bfill").values

        # Apply HMM to Each Time Series Window
        base_model = hmm.GaussianHMM(n_components=2, covariance_type='diag', min_covar=0.0001, startprob_prior=1.0,
                                 transmat_prior=1.0, means_prior=0, means_weight=0, covars_prior=0.03,
                                 covars_weight=1.5, algorithm='viterbi', random_state=42,
                                 n_iter=50, tol=0.001, verbose=False, params='stmc', init_params='stmc')
        windows1 = list(np.arange(0, len(signal_denoised), expected_iti * 5)) + [len(signal_denoised)]
        windows2 = list(np.arange(0, len(signal_denoised), expected_iti)) + [len(signal_denoised)]
        all_preds = []
        for windows in [windows1, windows2]:
            state_preds = []
            for w_start, w_end in zip(windows[:-1],windows[1:]):
                window_sig = signal_denoised[w_start:w_end]
                window_sig = max_window_normalization(window_sig, len(window_sig))
                # window_sig = preprocessing.MinMaxScaler((0,1)).fit_transform(window_sig.reshape(-1,1)).T[0]
                wmod = copy.copy(base_model)
                wmod.fit(window_sig.reshape(-1,1))
                wpred = wmod.predict(window_sig.reshape(-1,1))
                active_state = wpred[np.argmax(window_sig)]
                wpred = list(map(lambda i: 1 if i == active_state else 0, wpred))
                state_preds.append(wpred)
            state_preds = np.array(flatten(state_preds))
            all_preds.append(state_preds)
        state_preds = np.array(all_preds).T.max(axis = 1)

        # Smooth Binary
        state_len = 0
        current_state = state_preds[0]
        min_state_len = int(expected_iti / 3)
        smoothed_state_preds = [state_preds[0]]
        for s in state_preds:
            if s != current_state and state_len >= min_state_len:
                smoothed_state_preds.append(s)
                current_state = s
                state_len = 0
            elif s != current_state and state_len < min_state_len:
                smoothed_state_preds.append(current_state)
                state_len += 1
            else:
                smoothed_state_preds.append(s)
                state_len += 1
        state_preds = np.array(smoothed_state_preds)

        # Impact Detection using Estimated States
        statelens, statepos, states = rle(state_preds)
        win_start = statepos[np.nonzero(states)[0]]
        win_stop =  statepos[np.nonzero(states)[0]] + statelens[np.nonzero(states)[0]]
        impact_windows = list(zip(win_start, win_stop))
        state_preds_detected = np.array([find_tap_initiation(signal, int(statepos[t]),
                                    window_start=impact_windows[t][0]-20,window_end=impact_windows[t][1]+20)
                                     for t in range(len(impact_windows))])
        if state_preds_detected[-1] < state_preds_detected[-2]:
            state_preds_detected = state_preds_detected[:-1]
        state_preds_detected = np.unique(state_preds_detected)

        # Inter Tap Intervals
        met_ends = np.nonzero(state_preds_detected > metronome_beats.max())[0].min()
        met_itis, nomet_itis = np.diff(state_preds_detected[:met_ends+1]), np.diff(state_preds_detected[met_ends+1:])

        # Plot Results
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
    if (sub + 1) % 10 == 0 :
        print("Saving data")
        with open(stage_2_processed_tap_file,"wb") as the_file:
            pickle.dump(stage_2_processed, the_file, protocol = 2)

# Complete Save
print("Saving data")
with open(stage_2_processed_tap_file,"wb") as the_file:
    pickle.dump(stage_2_processed, the_file, protocol = 2)



###########################################################################################

# # Moving Max Normalization (due to baseline fluctuations)
# taps_per_window = 2
# normalized_signal = max_window_normalization(signal_denoised, expected_iti * taps_per_window)

# # Stabilize Predictions
# statelens, statepos, states = rle(state_preds)
# spans = pd.DataFrame(data = [statelens,statepos,states], index=["len","pos","state"]).T
# spans = spans.loc[spans.state == 1]
# spans = spans.loc[abs(spans.len - spans.len.median()) < 5 * spans.len.std()]
# state_changes = spans.pos.values

# # Filter by minimum intialization delay
# min_delay = .5 * expected_iti
# state_changes_filtered = [state_changes[0]]
# last_state_change = state_changes[0]
# for i in state_changes[1:]:
#     if i - last_state_change > min_delay:
#         state_changes_filtered.append(i)
#         last_state_change = i
# state_changes = np.array(state_changes_filtered)
