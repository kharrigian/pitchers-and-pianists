
###############################
### Imports
###############################

# Warning Supression
import warnings
warnings.simplefilter("ignore")

# Standard I/O and Data Handling
import pandas as pd
import numpy as np
import copy

# Signal Processing
import statsmodels.api as sm
from sklearn import preprocessing

# Data Loading
from scripts.libraries.helpers import rle, flatten

# Modeling
from hmmlearn import hmm
import peakutils.peak as peakutils

###############################
### Tap Processing Functions
###############################

# Normalization of signal between 0 and 1 (within windows)
def min_max_normalization(signal):
    """
    Normalize signal between 0 and 1 using sklearn MinMaxScaler

    Args:
        signal (array): Data to normalize

    Returns:
        normalized signal
    """
    return preprocessing.MinMaxScaler((0,1)).fit_transform(signal.reshape(-1,1)).T[0]

# Parameterize base HMM
base_model = hmm.GaussianHMM(n_components=2, covariance_type='diag', min_covar=0.0001,
                         transmat_prior=1.0, means_prior=0, means_weight=0, covars_prior=0.03,
                         covars_weight=1.5, algorithm='viterbi', random_state=42,
                         n_iter=100, tol=0.001, verbose=False, params='stmc', init_params='stmc')

# HMM-Based Tap Estimation (First Pass)
def hmm_first_pass(signal, expected_iti = 1000):
    """
    Use HMM on first pass to estimate location of taps

    Args:
        signal (array): Data to find tap initiations in
        expected_iti (int): Expected number of samples between successive peaks

    Returns:
        state_preds_detected (array): Locations of tap initiations
    """
    # Create Differently Sized Windows based on Expected ITI
    windows_large_range = list(np.arange(0, len(signal), expected_iti * 5)) + [len(signal)]
    windows_small_range = list(np.arange(0, len(signal), expected_iti * 2)) + [len(signal)]
    windows_global_range = [0, len(signal)]
    # Fit and Predict HMM within each window
    all_preds = []
    for windows in [windows_small_range, windows_large_range, windows_global_range]:
        state_preds = []
        for w_start, w_end in zip(windows[:-1],windows[1:]):
            # Select force within window
            window_sig = signal[w_start:w_end]
            # Normalize Between 1-100
            window_sig = np.floor(min_max_normalization(window_sig) * 100)
            # Copy Base HMM Model
            window_hmm = copy.copy(base_model)
            # Fit Model
            window_hmm.fit(window_sig.reshape(-1,1))
            # Predict States
            try:
                wpred = window_hmm.predict(window_sig.reshape(-1,1))
                active_state = wpred[np.argmax(window_sig)]
                wpred = list(map(lambda i: 1 if i == active_state else 0, wpred))
            except:
                wpred = [0 for i in window_sig]
            # Add States
            state_preds.append(wpred)
        # Flatten and Update State Store
        state_preds = np.array(flatten(state_preds))
        all_preds.append(state_preds)
    # Require All Scales to Predict and Active State to Reduce False Positives
    state_preds = (np.array(all_preds).T.sum(axis = 1) == 3).astype(int)
    # Impact Detection using Estimated States
    statelens, statepos, states = rle(state_preds)
    state_preds_detected = statepos[np.nonzero(states)[0]]
    return state_preds_detected

# Peaktils-Based First Pass
def findpeaks_first_pass(signal, expected_iti = 1000, percentile_thres = 90):
    """
    Use standard peak detection to find peaks during first pass

    Args:
        signal (array): Data to find peaks within
        expected_iti (int): Number of samples expected between peaks
        percentile_thres (int [0,100]): Percentile to use as minimum detection threshold

    Returns:
        peaks, locations of peaks in the signal
    """
    # Normalize Between 0-1
    signal_normed = min_max_normalization(signal) ## Find Peaks
    # Compute Threshold
    threshold = np.percentile(signal_normed, percentile_thres)
    # Find Peaks
    peaks = peakutils.indexes(signal_normed, thres = threshold, min_dist = expected_iti / 3, thres_abs = True)
    return peaks

# Second Pass Tap Initiation
def estimate_tap_initiation(signal, estimated_locations, expected_iti = 1000):
    """
    Zero in on the true tap initiations from original prediction using HMMs again

    Args:
        signal (array): Signal to use for fitting the HMM
        estimated_locations (array): Estimated tap initiations from first pass
        expected_iti (int): Expected distance between peaks. Default is 1000
    """
    window_edges = [int(max([estimated_locations[0] - .2*expected_iti,0]))] + \
                    [int(np.mean(estimated_locations[i:i+2])) for i in range(len(estimated_locations)-1)] + \
                    [int(min([estimated_locations[-1] + .2*expected_iti,len(signal)-1]))]
    filtered_inits = []
    for start, stop, est in zip(window_edges[:-1],window_edges[1:], estimated_locations):
        signal_window = signal[start:stop]
        signal_window_normalized = min_max_normalization(signal_window)
        # clean the signal
        decomposed = sm.tsa.seasonal_decompose(signal_window_normalized, model = "additive", freq = 10)
        signal_denoised = pd.Series(decomposed.trend).fillna(method = "ffill").fillna(method="bfill").values
        # remodel around nice window
        mod = copy.copy(base_model)
        try:
            mod.fit(signal_denoised.reshape(-1,1))
            pred = mod.predict(signal_denoised.reshape(-1,1))
        except:
            continue
        active_state = pred[np.argmax(signal_denoised)]
        pred = list(map(lambda p: 1 if p == active_state else 0, pred))
        # extract state changes
        statelens, statepos, states = rle(pred)
        state_preds_detected, statelens_detected = statepos[np.nonzero(states)[0]], statelens[np.nonzero(states)[0]]
        # Require Monotonic Increasing/Constant
        increasing = pd.Series(signal_denoised).diff().fillna(0) > 0
        state_preds_detected = [s for s in state_preds_detected if increasing[s-5:s+6].sum() > 3]
        # Append Guesses
        if len(state_preds_detected) > 0:
            # Might be multiple
            estimates = [min(state_preds_detected)]
            last_estimate = estimates[0]
            for s in state_preds_detected[1:]:
                if s - last_estimate > (expected_iti/3):
                    estimates.append(s)
                    last_estimate = s
            # Append all estimates
            for e in estimates:
                filtered_inits.append(start + e)
    return np.array(filtered_inits)

# Wrapper for Peak Detection
def find_taps(signal, expected_intertapinterval = 1000, use_hmm = True, peakutils_threshold = 90):
    """
    Find taps within signal

    Args:
        signal (array): The digital force signal
        expected_intertapinterval (int): Number of samples to use as the expected distance between peaks
                                         Default is 1000 (500ms at 2000hz)
        use_hmm (bool): Whether to use the HMM-based method. If False, relies on peakutils
        peakutils_threshold (int [0,100]): If using peakutils method, percentile of data to use as minimum height

    Returns:
        state_preds_detected (array): Estimated tap initiation locations
    """
    # Extract Main Trend in Signal
    decomposition = sm.tsa.seasonal_decompose(signal, model = "additive", freq = 50)
    signal_denoised = pd.Series(decomposition.trend).fillna(method = "ffill").fillna(method="bfill").values
    # Quantize Signal (base = rounding integer)
    quantizer = lambda x, base: int(base * round(float(x)/base))
    signal_quantized = np.array([quantizer(f, 2) for f in (min_max_normalization(signal_denoised) * 100)])
    # Estimate Peaks
    if use_hmm:
        state_preds_detected = hmm_first_pass(signal_quantized, expected_intertapinterval)
    else:
        state_preds_detected = findpeaks_first_pass(signal_quantized, expected_intertapinterval, percentile_thres = peakutils_threshold)
    # Require Monotonic Increasing/Constant (Not applicable for peaks)
    if use_hmm:
        increasing = pd.Series(signal_quantized).diff().fillna(0) > 0
        state_preds_detected = [s for s in state_preds_detected if increasing[s-5:s+6].sum() > 0]
    # Filter out double taps
    min_delay = expected_intertapinterval / 3
    last_tap = 0
    filtered_preds_detected = []
    for pred in state_preds_detected:
        if pred - last_tap >= min_delay:
            filtered_preds_detected.append(pred)
            last_tap = pred
    state_preds_detected = np.array(filtered_preds_detected)
    # Get as close to tap initiation as possible using HMMs
    if use_hmm:
        state_preds_detected = estimate_tap_initiation(signal_denoised, state_preds_detected, expected_intertapinterval)
    return state_preds_detected
