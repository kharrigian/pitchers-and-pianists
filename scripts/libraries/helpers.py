
########################
### Imports
########################

from __future__ import print_function, division  # requires python 2.6 or later (2.7 or later preferred)
import math

# Standard I/O and Data Handling
import pandas as pd
import numpy as np
import scipy.io as sio
import pickle

# Warning Supression
import warnings
warnings.simplefilter("ignore")

########################
### Functions
########################

# Flatten a list of lists
flatten = lambda l: [item for sublist in l for item in sublist]

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

# Load a pickle file
def load_pickle(filename, how = "rb"):
    with open(filename, how) as the_pickle_file:
        return pickle.load(the_pickle_file)

# Load a pickle file
def dump_pickle(pyobject, filename, how = "wb"):
    with open(filename, how) as the_pickle_file:
        pickle.dump(pyobject, the_pickle_file, protocol = 2)

########################
### Composite Standard Deviation
########################

__all__ = ['iterable', 'avg', 'sample_SD', 'composite_SD']

def iterable(obj):
    '''True iff obj is iterable: a list, tuple, or string.'''
    return hasattr(obj, '__contains__')

def avg(samples):
    if len(samples) >= 1:
        return sum(samples) / len(samples)
    return float('nan')

def sample_SD(samples):
    '''input is an array of samples; result is the standard deviation'''
    mean = avg(samples)
    sum_of_squared_deviations = 0;
    sd = 0
    if len(samples) >= 2:
        for datum in samples:
            sum_of_squared_deviations += ((datum - mean) * (datum - mean));
        sd = math.sqrt(sum_of_squared_deviations / (len(samples)-1) );
    return sd

def composite_SD(means, SDs, ncounts):
    '''Calculate combined standard deviation via ANOVA (ANalysis Of VAriance)
       See:  http://www.burtonsys.com/climate/composite_standard_deviations.html
       Inputs are:
         means, the array of group means
         SDs, the array of group standard deviations
         ncounts, number of samples in each group (can be scalar
                  if all groups have same number of samples)
       Result is the overall standard deviation.
    '''
    G = len(means)  # number of groups
    if G != len(SDs):
        raise Exception('inconsistent list lengths')
    if not iterable(ncounts):
        ncounts = [ncounts] * G  # convert scalar ncounts to array
    elif G != len(ncounts):
        raise Exception('wrong ncounts list length')

    # calculate total number of samples, N, and grand mean, GM
    N = sum(ncounts)  # total number of samples
    if N <= 1:
        raise Exception("Warning: only " + str(N) + " samples, SD is incalculable")
    GM = 0.0
    for i in range(G):
        GM += means[i] * ncounts[i]
    GM /= N  # grand mean

    # calculate Error Sum of Squares
    ESS = 0.0
    for i in range(G):
        ESS += ((SDs[i])**2) * (ncounts[i] - 1)

    # calculate Total Group Sum of Squares
    TGSS = 0.0
    for i in range(G):
        TGSS += ((means[i]-GM)**2) * ncounts[i]

    # calculate standard deviation as square root of grand variance
    result = math.sqrt((ESS+TGSS)/(N-1))
    return result
