
###############################################################################################################
### Setup
###############################################################################################################

##################
### Imports
##################

# Standard
import pandas as pd
import numpy as np
import os

# Statistical Modeling
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Helpers
from scripts.libraries.helpers import flatten

##################
### Helpers
##################

## Assign Value to a Bin
assign_bin = lambda val, bin_array: [b for b in range(len(bin_array)-1) if val >= bin_array[b] and val <= bin_array[b+1]][0]
quantiles = [0,.067,.159,.309,.50,.691,.841,.933, 1] # Based on 3-Standard Deviations

##################
### Variables
##################

"""
Time Definitions
-----------------
met_last10 & met_last5: Last 10 and last 5 seconds, respectively, of synchronization period (full time 15 seconds)
nomet_last20 & nomet_first5 & nomet_last5: 20-40 seconds, 20-25 seconds, and 35-40 seconds

Metric Definitions
------------------
cv: Standard Deviation / Median
sync_error: (Median - Trial ITI) / Trial ITI * 100
drift: (Median_last5 - Median_first5) / Median_first5 * 100

Trial Variables
-------------------
trial: Number of tapping trial (1-6)
speed_occurrence: Has the subject done this speed already? (1-2)
trial_speed: SlowedDown (120% preferred), NoChange (100% preferred), SpedUp (80% preferred)

Subject Variables
-------------------
age: Age of the subject in years
gender: Self-identified gender of the subject (1 = female, 0 = male)
musical_experience: Does the subject have any musical experience (1 = yes, 0 = no)
sport_experience: Does the subject have any sport experience (1 = yes, 0 = no)
musical_experience_yrs, sport_experience_yrs: Number of years subject has practiced that activity (-1 = specified activity, missing years)
"""

# Dependent Variables
cv_metrics = ["met_cv_last10","met_cv_last5","nomet_cv_last20", "nomet_cv_first5","nomet_cv_last5"]
error_metrics = ["met_sync_error_last10","met_sync_error_last5","nomet_sync_error_last20","nomet_sync_error_first5","nomet_sync_error_last5"]
drift_metrics = ["nomet_drift"]

# Variable Types
factor_variables = ["trial","speed_occurrence","trial_speed", "gender", "musical_experience", "sport_experience"]
continuous_variabless = ["preferred_period","age"]
all_variables = factor_variables + continuous_variabless

# Fixed Primary Variables
primary_variables = ["age","gender","trial","speed_occurrence","trial_speed","musical_experience","sport_experience"]

##################
### Load Results
##################

# Load Results
results = pd.read_csv("./data/processed_results.csv")

# Drop Rows with Null Primary Metrics
results = results.loc[results[cv_metrics + error_metrics + drift_metrics + primary_variables].isnull().sum(axis = 1) == 0]

# Drop Subjects without 6 Trials
results = results.loc[results.subject.isin(set(results.subject.value_counts().loc[results.subject.value_counts() == 6].index.values))].copy()

# Create Separate DataFrame For Subjects that "Experience" Testing
results_musical_experience = results.loc[(results.musical_experience_yrs >= 1)].copy()
results_sports_experience = results.loc[(results.sport_experience_yrs >= 1)].copy()

# Age Bins
age_bins = stats.mstats.mquantiles(results.drop_duplicates("subject")["age"].values, quantiles)
results["age_bin"] = results["age"].map(lambda val: assign_bin(val, age_bins))

##################
### Derived Values
##################

# Absolute Synchronization Error
for error in ["met_sync_error_last5","met_sync_error_last10","nomet_sync_error_first5","nomet_sync_error_last5","nomet_sync_error_last20"]:
    results["abs_%s" % error] = np.abs(results[error])
    results_musical_experience["abs_%s" % error] = np.abs(results_musical_experience[error])
    results_sports_experience["abs_%s" % error] = np.abs(results_sports_experience[error])

# Proprotional Experience
results_musical_experience["musical_experience_yrs_normalized"] = results_musical_experience["musical_experience_yrs"] / results_musical_experience["age"]
results_sports_experience["sport_experience_yrs_normalized"] = results_sports_experience["sport_experience"] / results_sports_experience["age"]

# Create Musical Experience Bins
music_bins = stats.mstats.mquantiles(results_musical_experience.drop_duplicates("subject")["musical_experience_yrs"].values, quantiles)
music_normed_bins = stats.mstats.mquantiles(results_musical_experience.drop_duplicates("subject")["musical_experience_yrs_normalized"].values, quantiles)
results_musical_experience["musical_experience_bin"] = results_musical_experience.musical_experience_yrs.map(lambda val: assign_bin(val, music_bins))
results_musical_experience["musical_experience_normalized_bin"] = results_musical_experience.musical_experience_yrs_normalized.map(lambda val: assign_bin(val, music_normed_bins))

# Create Sport Experience Bins
sport_bins = stats.mstats.mquantiles(results_sports_experience.drop_duplicates("subject")["sport_experience_yrs"].values, quantiles)
sport_normed_bins = stats.mstats.mquantiles(results_sports_experience.drop_duplicates("subject")["sport_experience_yrs_normalized"].values, quantiles)
results_sports_experience["sport_experience_bins"] = results_sports_experience.sport_experience_yrs.map(lambda val: assign_bin(val, sport_bins))
results_sports_experience["sport_experience_normalized_bin"] = results_sports_experience.sport_experience_yrs_normalized.map(lambda val: assign_bin(val, sport_normed_bins))

###############################################################################################################
### Analysis
###############################################################################################################

##################
### Syncronization Error
##################

##### Synchronization Error Metric #####
sync_error = "abs_met_sync_error_last10"
alpha = 1e-5 # Ensure nonzero in log
results["sync_error_transformed"] = np.log(results[sync_error] + alpha)

## Dependent Variable Distribution
fig, ax = plt.subplots(1,2, figsize = (10,5), sharey=True)
ax[0].hist(results[sync_error], bins = 25, normed = True)
ax[1].hist(results["sync_error_transformed"], bins = 25, normed = True)
ax[0].set_title(sync_error)
ax[1].set_title("%s log-transformed" % sync_error)
ax[0].set_ylabel("Frequency")
fig.tight_layout()
plt.show()

##################
### Continuation Error
##################

##### Continuation Error Metric #####
unpaced_error = "abs_nomet_sync_error_last20"
alpha = 1e-5 # Ensure nonzero in log
results["cont_error_transformed"] = np.log(results[unpaced_error] + alpha)

## Dependent variable distribution
fig, ax = plt.subplots(1,2, figsize = (10,5), sharey=True)
ax[0].hist(results[unpaced_error], bins = 25, normed = True)
ax[1].hist(results["cont_error_transformed"], bins = 25, normed = True)
ax[0].set_title(unpaced_error)
ax[1].set_title("%s log-transformed" % unpaced_error)
ax[0].set_ylabel("Frequency")
fig.tight_layout()
plt.show()

##################
### Variability
##################

##### Variability (CV) Metric #####
variability_met = "met_cv_last10"
variability_cont = "nomet_cv_last20"
alpha = 1e-5
results["var_met_transformed"] = np.log(results[variability_met] + alpha)
results["var_nomet_transformed"] = np.log(results[variability_cont] + alpha)

## Transformed Distribution
fig, ax = plt.subplots(2,2)
ax[0][0].hist(results[variability_met], bins = 25)
ax[1][0].hist(results[variability_cont], bins = 25)
ax[0][1].hist(results["var_met_transformed"], bins = 25)
ax[1][1].hist(results["var_nomet_transformed"], bins = 25)
for i, itit in enumerate([variability_met, variability_cont]):
    for j, jtit in enumerate(["","log-transformed"]):
        ax[i][j].set_title(("%s %s" % (itit, jtit)).rstrip())
plt.tight_layout()
plt.show()

##################
### Drift
##################

##### Drift Metric #####
drift = "nomet_drift"

## Dependent Variable Distribution (Note that drift is normally distributed, no transformation needed)
fig, ax = plt.subplots(1,1, figsize = (10,5), sharey=False)
ax.hist(results[drift], bins = 25, normed = True)
ax.set_title(drift)
ax.set_ylabel("Frequency")
fig.tight_layout()
plt.show()


##################
### Subset of Results
##################

cols = ["subject","trial","trial_speed","speed_occurrence","age","gender","musical_experience","sport_experience",
            sync_error, "sync_error_transformed", unpaced_error, "cont_error_transformed",
            variability_met, "var_met_transformed",variability_cont, "var_nomet_transformed", drift]
results[cols].to_csv("./data/transformed_subset.csv")

# ## Summarize Demographics of Subject Pool
# def summarize_subject_pool(results_data):
#     results_data = results_data.copy()
#     results_data = results_data.drop_duplicates(["subject"])
#     n_subjects = len(results_data)
#     age_stats = results_data.age.mean(), results_data.age.std(), results_data.age.min(), results_data.age.max()
#     percent_female = results_data.gender.value_counts(normalize=True)[1] * 100
#     percent_musical_experience = results_data.musical_experience.value_counts(normalize=True)[1] * 100
#     percent_sport_experience = results_data.sport_experience.value_counts(normalize=True)[1] * 100
#     return {"Subjects":n_subjects, "Age":dict(zip(["Mean","Std","Min","Max"], age_stats)),
#                 "PercentFemale":percent_female, "PercentMusicalExperience":percent_musical_experience,
#                 "PercentSportExperience":percent_sport_experience}

# # Summarize Pools
# results_pool_demos = summarize_subject_pool(results)
# results_musical_pool_demos = summarize_subject_pool(results_musical_experience)
# results_sports_pool_demos = summarize_subject_pool(results_sports_experience)
