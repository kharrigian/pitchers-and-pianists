
## Analysis of Study

################################################################################
### Setup -- Data Loading and Cleaning
################################################################################

###############
### Imports
###############

# Warning Supression
import warnings
warnings.simplefilter("ignore")

# Standard
import pandas as pd
import numpy as np
import os, sys

# Statistical Modeling/Curve Fitting
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from scipy.optimize import curve_fit

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from matplotlib import rcParams

# Helpers
from scripts.libraries.helpers import load_pickle

###############
### Plot Helpers
###############

## Plotting Variables
standard_fig = (10,5.8)
plot_dir = "./plots/"
stats_plots = plot_dir + "analysis/"
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Helvetica']
rcParams["errorbar.capsize"] = 5

## Plot subdirectories
for d in [plot_dir, stats_plots]:
    if not os.path.exists(d):
        os.mkdir(d)

###############
### General Helpers
###############

## Create Equal Percentiles
create_percentiles = lambda n_bins: np.linspace(0,100, n_bins+1)

## Using Percentiles, Estimate Equal Bin Ranges on a set of Values
create_equal_bins = lambda values, n_bins: np.percentile(values, create_percentiles(n_bins))

## Standard Error
std_error = lambda vals: np.std(vals) / np.sqrt(len(vals) - 1)

## Assign Value to a Bin
assign_bin = lambda val, bin_array: np.max([b for b in range(len(bin_array)) if val >= bin_array[b]])
create_bin_strings = lambda values, bins: ["[%d,%d)" % (np.ceil(x),np.ceil(y)) for x,y in list(zip(bins[:-1], bins[1:]))]+ ["[%d,%d]" % (bins[-1], max(values))]

## Flatten a List of Lists
flatten = lambda l: [item for sublist in l for item in sublist]

## Holm-Bonferroni Correction
def apply_bonferroni_correction(df, pval_col, alpha = 0.05):
    """
    Apply Holm-Bonferroni correction to Wald Test statistics
    """
    comparisons = [a for a in df.index.values if a not in ["Intercept", "groups RE"]]
    df = df.replace("",np.nan).dropna()
    pvals = df.loc[comparisons][[pval_col,"df_constraint"]].astype(float)
    n_comparisons = pvals.df_constraint.sum()
    pvals = pvals.sort_values(pval_col,ascending=True)
    pvals["bonferroni_max"] = (alpha / (1 + n_comparisons - pvals["df_constraint"].cumsum()))
    pvals["significant"] = list(map(lambda i: "*" if i else "", (pvals[pval_col] < pvals["bonferroni_max"]).values))
    df = pd.merge(df, pvals[["significant"]], how = "left", left_index=True, right_index=True)
    df["significant"] = df["significant"].fillna("")
    return df

## Mixed LM Model Summary
def summarize_lm_model(model, alpha = 0.05):
    ## Model Params
    model_fit = model.summary().tables[0]
    model_fit.columns = [""] * 4
    ## Model Statistics
    model_summary = model.summary().tables[1]
    wald_test = apply_bonferroni_correction(model.wald_test_terms().table, "pvalue", alpha)
    wald_test.rename(columns = {"statistic":"chi_squared"},inplace=True)
    ## Display
    return model_summary, wald_test

## Wald Significant Values
def print_significant(model, wald_stats):
    """
    Print out significant effects
    """
    sample_size = int(model.summary().tables[0].set_index(0).loc["No. Groups:", 1])
    sig_effects = wald_stats.loc[wald_stats.significant == "*"].copy()
    for factor in sig_effects.index:
        chi2 = wald_stats.loc[factor]["chi_squared"]
        p = wald_stats.loc[factor]["pvalue"].item()
        df = wald_stats.loc[factor]["df_constraint"]
        outstring = "- {} chi^2({}, N = {}) = {:.5f}, p = {:.5f}".format(factor, df, sample_size, chi2, p )
        print(outstring)

## Bootstrapped Confidence Interval
def bootstrap_ci(values, alpha = 0.05, func = np.mean, sample_percent = 20, samples = 1000):
    processed_vals = []
    values = np.array(values)
    for sample in range(samples):
        sample_vals = np.random.choice(values, int(values.shape[0] * sample_percent/100.), replace = True)
        processed_vals.append(func(sample_vals))
    return np.percentile(processed_vals, [alpha*100/2, 50, 100. - (alpha*100/2)])

## Exponential Fit
class ExponentialFit:

    def __init__(self):
        self.exp_model = lambda x, a, b, c: a**(b * x) + c
        self.is_fit = False

    def fit(self, x, y):
        self.opt_params, self.cov_params = curve_fit(self.exp_model, x, y)
        self.is_fit = True

    def predict(self, x):
        if not self.is_fit:
            raise Exception("Exponential model has not yet been fit")
        return self.exp_model(x, self.opt_params[0], self.opt_params[1], self.opt_params[2])

    def fit_and_plot_scatter(self, x, y):
        self.fit(x, y)
        x_unique = np.sort(np.unique(x))
        fig, ax = plt.subplots(figsize = standard_fig)
        ax.scatter(x, y, alpha = 0.5, color = "blue", s = 10)
        ax.plot(x_unique, self.predict(x_unique), color = "orange", linestyle = "--", linewidth = 2)
        return fig, ax

###############
### Global Variables
###############

# CV Variables
cv_sync_metrics = ["met_cv_last5"]
cv_cont_metrics = ["nomet_cv_last5"]
# cv_cont_metrics = ["nomet_cv_first5","nomet_cv_last5"]
cv_metrics = cv_sync_metrics + cv_cont_metrics

# Error Variables
error_sync_metrics = ["met_sync_error_last5", "met_sync_error_last5_rel"]
error_cont_metrics = ["nomet_sync_error_last5","nomet_sync_error_last5_rel"]
# error_cont_metrics = ["nomet_sync_error_first5","nomet_sync_error_last5","nomet_sync_error_first5_rel","nomet_sync_error_last5_rel"]
error_metrics = error_sync_metrics + error_cont_metrics

# Drift Variables
drift_metrics = ["nomet_drift","nomet_drift_rel", "nomet_drift_regression"]

# Variable Types
factor_variables = ["trial","speed_occurrence","trial_speed", "gender", "musical_experience", "sport_experience"]
continuous_variabless = ["preferred_period","age"]
all_variables = factor_variables + continuous_variabless

# Fixed Primary Variables
primary_variables = ["age","gender","trial","speed_occurrence","trial_speed","musical_experience","sport_experience"]

###############
### Load and Filter Data
###############

# Load Results (file generated by `extract_tap_metrics.py`)
results = pd.read_csv("./data/processed_results.csv")

# Drop Rows with Null Primary Metrics
results = results.loc[results[cv_metrics + error_metrics + drift_metrics + primary_variables].isnull().sum(axis = 1) == 0]

# Drop Subjects without 6 Trials
results = results.loc[results.subject.isin(set(results.subject.value_counts().loc[results.subject.value_counts() == 6].index.values))].copy()

# Add Age Bins (Explict Choice to Align with "Pitchers" study)
age_bins = [5,10,20,30,50,69]
age_bin_points = [7.5, 15, 25, 40, 59]
age_bin_strings = ["5-9","10-19","20-29","30-49","50+"]
results["age_bin"] = results["age"].map(lambda val: assign_bin(val, age_bins))

###############
### Add Disorder Flag
###############

## Specify Disorders to Flag
disorder_filters = ["BROKEN WRIST","CARPAL TUNNEL","ACUTE ARTHRITIS","BROKEN WRIST (2 YEARS AGO)",
                   "AUTISM","ADHD","TENDONITIS","BROKEN ARMS, BACK, ANKLE", "ARTHRITIS",
                   "COMPLEX REGIONAL PAIN SYNDROME", "TICK DISORDER", "FIBROMYALGIA",
                   "CARPAL TUNNEL (BILATERAL)", "SLIGHT GRIP PAIN (NONE DURING EXPERIMENT)",
                   "'AGENESIS OF THE CORPUS COLLOSUM'","CONNECTIVE TISSUE DISORDER",
                   "CERVICAL FUSION (NECK)","MALLET FINGER"]

## Remove Disordered Subjects
results["healthy"] = np.logical_not(results.specify_disorder.isin(disorder_filters))

###############
### Absolute Values
###############

# Absolute Synchronization Error (For when we don't care about directional differences, just absolute)
for error in error_metrics + drift_metrics:
    results["abs_%s" % error] = np.abs(results[error])

# Update Variable Groups
error_metrics = error_metrics + ["abs_{}".format(met) for met in error_metrics]
drift_metrics = drift_metrics + ["abs_{}".format(met) for met in drift_metrics]

###############
### Account for Trial/Block Effects
###############

## Average Metrics Across Both Trials for a Given Trial Speed
metrics_to_average = error_metrics + cv_metrics + drift_metrics
mean_results = pd.pivot_table(index = ["subject","trial_speed"], values = metrics_to_average, aggfunc=np.mean,
                             data = results).reset_index()
merge_vars = ["subject","age","age_bin","gender","musical_experience","musical_experience_yrs",
              "sport_experience","sport_experience_yrs", "preferred_period","healthy"]
mean_results = pd.merge(mean_results, results.drop_duplicates("subject")[merge_vars], left_on = "subject", right_on = "subject")

################################################################################
### Independent Variable Distributions
################################################################################

## De-duplicate the data set based on subject (only want unique characteristics)
subject_deduped = results.drop_duplicates("subject")

###############
### Preferred Period
###############

## Preferred Period Distribution
mean_pp, std_pp = subject_deduped.preferred_period.mean(), subject_deduped.preferred_period.std()
counts, bins = np.histogram(subject_deduped["preferred_period"], 20)
fig, ax = plt.subplots(1,1, figsize = standard_fig, sharey = False, sharex = False)
b = ax.hist(subject_deduped["preferred_period"], bins = bins, normed = False, color = "blue", edgecolor = "navy", alpha = .5)
ax.set_xlabel("Preferred Period (ms)", fontsize = 18, fontweight = "bold")
ax.set_ylabel("Subjects", fontsize = 18, fontweight = "bold")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize=16)
fig.tight_layout()
plt.savefig(stats_plots + "preferred_period")
plt.close()

## No Correlation between age and preferred period
fig, ax = plt.subplots(1,1, figsize = standard_fig, sharey = False, sharex = False)
ax.scatter(subject_deduped["age"], subject_deduped["preferred_period"], color = "blue", edgecolor="navy", alpha = .5,
            s = 50)
ax.set_xlabel("Age", fontsize = 18, fontweight = "bold")
ax.set_ylabel("Preferred period (ms)", fontsize = 18, fontweight = "bold")
ax.tick_params(labelsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout()
plt.savefig(stats_plots + "age_preferred_period")
plt.close()

###############
### Age & Gender
###############

## Replace Binary Gender Forms
results["gender"] = results["gender"].map(lambda g: "Female" if g == 1 else "Male")
subject_deduped["gender"] = subject_deduped["gender"].map(lambda g: "Female" if g == 1 else "Male")

## Plot Age + Gender Distribution
fig, ax = plt.subplots(figsize = standard_fig)
max_count = 0
for a, age in enumerate(subject_deduped.age_bin.unique()):
    for g, gender in enumerate(["Female","Male"]):
        demo_count = len(subject_deduped.loc[(subject_deduped.age_bin == age)&(subject_deduped.gender==gender)])
        ax.bar(0.025 + g*0.45 + a, demo_count, color = {"Male":"blue","Female":"red"}[gender],
        alpha = .5, align = "edge", width = .45, label = gender if a == 0 else "",
        edgecolor = {"Male":"navy","Female":"darkred"}[gender])
        if demo_count > 0:
            ax.text(0.25 + g*0.45 + a, demo_count + 1, demo_count, ha = "center", fontsize = 18)
            max_count = demo_count if demo_count > max_count else max_count
ax.legend(loc = "upper right", frameon = True, fontsize = 18)
ax.set_xticks(np.arange(a+1)+.5)
ticks = ax.set_xticklabels(age_bin_strings, rotation = 0)
ax.set_xlabel("Age", fontsize = 18, fontweight = "bold")
ax.set_ylabel("Subjects", fontsize = 18, fontweight = "bold")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
lim = ax.set_ylim(0, max_count+5)
ax.tick_params(labelsize = 16)
fig.tight_layout()
fig.savefig(stats_plots + "demographics.png", dpi=300)
plt.close()

## Age + Gender + Musical Experience
fig, ax = plt.subplots(figsize = standard_fig)
for a, age in enumerate(subject_deduped.age_bin.unique()):
    for g, gender in enumerate(["Female","Male"]):
        bottom = 0
        for e, experience in enumerate([1,0]):
            demo_count = len(subject_deduped.loc[(subject_deduped.age_bin == age)&(subject_deduped.gender==gender)&(subject_deduped.musical_experience==experience)])
            ax.bar(0.025 + g*0.45 + a, demo_count, bottom = bottom, color = {"Male":"blue","Female":"red"}[gender],
            alpha = 0.25 if e == 1 else .6, align = "edge", width = .45,
            label = "{} ({})".format(gender, {1:"w/ M.E.",0:"w/o M.E."}[experience]) if a == 0 else "",
            edgecolor = {"Male":"navy","Female":"darkred"}[gender])
            bottom += demo_count
ax.legend(loc = "upper right", frameon = True, fontsize = 18)
ax.set_xticks(np.arange(a+1)+.5)
ticks = ax.set_xticklabels(age_bin_strings, rotation = 0)
ax.set_xlabel("Age", fontsize = 18, fontweight = "bold")
ax.set_ylabel("Subjects", fontsize = 18, fontweight = 'bold')
ax.tick_params(labelsize = 16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
fig.tight_layout()
fig.savefig(stats_plots + "demographics_musicalexperience.png", dpi=300)
plt.close()

###############
### Musical Experience
###############

## Musical Experience Distribution
musical_experience_check = lambda row: "No Musical Experience" if row["musical_experience"] == 0 else \
                                                "Has Musical Experience\n(Known Amount)" if row["musical_experience_yrs"] >= 1 else \
                                                "Has Musical Experience \n(Unknown Amount)"
subject_deduped["musical_experience_specified"] = subject_deduped.apply(musical_experience_check, axis = 1)
musical_experience_dist = subject_deduped.musical_experience_specified.value_counts()
music_exp_subset = subject_deduped.loc[subject_deduped.musical_experience_yrs >= 1]
fig, ax = plt.subplots(1,2, figsize = standard_fig)
musical_experience_dist.plot.barh(ax = ax[0], color = "blue", edgecolor = "navy", alpha = .5)
ax[0].set_yticks(np.arange(3)); ax[0].set_yticklabels(musical_experience_dist.index.values, multialignment = "center")
ax[1].scatter(music_exp_subset["age"], music_exp_subset["musical_experience_yrs"], color = "blue", edgecolor = "navy", s = 50, alpha = .5)
ax[1].plot([0, music_exp_subset.age.max()], [0, music_exp_subset.age.max()], linestyle = "--", color = "navy", alpha = .5)
ax[0].set_xlabel("Subjects",fontsize=18, fontweight = "bold")
ax[1].set_xlabel("Age",fontsize=18, fontweight = "bold"); ax[1].set_ylabel("Years of Musical Experience", labelpad = 10, fontsize=18, fontweight = "bold")
for a in ax:
    a.tick_params(labelsize = 16)
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
fig.tight_layout()
fig.subplots_adjust(wspace = .4)
plt.savefig(stats_plots + "musical_experience")
plt.close()

## Musical Experience with Age/Gender
max_gender_count = 0; max_age_count = 0
fig, ax = plt.subplots(1,2, figsize = standard_fig)
for x, experience in enumerate([0,1]):
    for g, gender in enumerate(["Female","Male"]):
        demo_counts = len(subject_deduped.loc[(subject_deduped.musical_experience==experience)&
                                             (subject_deduped.gender==gender)])
        ax[0].bar(0.025 + x*.45 + g, demo_counts, color = {0:"orange",1:"blueviolet"}[experience],
                  alpha = .5, label = {1:"Has Musical Experience",0:"No Musical Experience"}[experience] if g == 0 else "",
                  width = .45, align = "edge", edgecolor = "darkviolet" if x == 1 else "darkorange")
        ax[0].text(0.025+ x*.45 + .45/2 + g, demo_counts + 4, demo_counts + 1, ha="center", fontsize = 16)
        max_gender_count = demo_counts if demo_counts > max_gender_count else max_gender_count

    for a, age in enumerate(subject_deduped.age_bin.unique()):
        demo_counts = len(subject_deduped.loc[(subject_deduped.musical_experience==experience)&
                                             (subject_deduped.age_bin==age)])
        ax[1].bar(0.025 + x*.45 + a, demo_counts, color = {0:"orange",1:"blueviolet"}[experience],
                  alpha = .5, label =  {1:"Has Musical Experience",0:"No Musical Experience"}[experience] if a == 0 else "",
                  width = .45, align = "edge", edgecolor = "darkviolet" if x == 1 else "darkorange")
        ax[1].text(0.025+ x*.45 +.45/2 + a, demo_counts + 1, demo_counts + 1, ha="center", fontsize = 16)
        max_age_count = demo_counts if demo_counts > max_age_count else max_age_count
ax[0].set_ylim(0, max_gender_count+20)
ax[1].set_ylim(0, max_age_count + 20)
ax[0].set_xticks(np.arange(2)+.5)
ax[0].set_xticklabels(["Female","Male"])
ax[1].set_xticks(np.arange(a+1)+.5)
ax[1].set_xticklabels(age_bin_strings, rotation = 0)
handles, labels = ax[0].get_legend_handles_labels()
leg = fig.legend(handles, labels, loc='upper center', ncol = 2, fontsize = 16)
for t in leg.texts:
    t.set_multialignment('center')
for a in ax:
    a.set_ylabel("Subjects", fontsize = 18, fontweight = "bold")
    a.tick_params(labelsize = 14)
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
ax[0].set_xlabel("Gender", fontsize = 18, fontweight = "bold")
ax[1].set_xlabel("Age", fontsize = 18, fontweight = "bold")
fig.tight_layout()
fig.subplots_adjust(top = .86)
plt.savefig(stats_plots + "musical_experience_demographics")
plt.close()

###############
### Make Amenable to Condition Modeling
###############

## Choose Metrics
sync_error = "abs_met_sync_error_last5_rel"
cont_error = "abs_nomet_sync_error_last5_rel"
sync_cv_metric = "met_cv_last5"
cont_cv_metric = "nomet_cv_last5"
drift_metric = "nomet_drift_rel"

## Separate Columns
standard_cols = ["subject","trial_speed","age","age_bin","gender","musical_experience","musical_experience_yrs",
            "sport_experience","sport_experience_yrs","preferred_period","healthy"]
met_cols = [sync_error, sync_cv_metric]
nomet_cols = [cont_error, cont_cv_metric, drift_metric]

## Separate DataFrames
met_df = mean_results[standard_cols + met_cols].rename(columns = {sync_error:"error",sync_cv_metric:"cv"}).copy()
nomet_df = mean_results[standard_cols + nomet_cols].rename(columns = {cont_error:"error",cont_cv_metric:"cv",
                                                                        drift_metric:"drift"}).copy()

## Add Condition Columns
met_df["condition"] = "paced"
nomet_df["condition"] = "unpaced"

## Concatenate DataFrames
merged_results_df = pd.concat([met_df, nomet_df], sort=True)
merged_results_df.index = np.arange(len(merged_results_df))

################################################################################
### Error Analysis
################################################################################

## Fit Mixed LM Model
error_model_formula = "error ~ C(gender) + age + preferred_period + trial_speed + C(musical_experience) + C(condition)"
error_model = smf.mixedlm(error_model_formula, data = merged_results_df, groups = merged_results_df["subject"]).fit(reml=True)
error_summary, error_wald = summarize_lm_model(error_model)

print("Error Effects:"); print_significant(error_model, error_wald)

"""
Error Effects:
- trial_speed chi^2(2, N = 303) = 12.05179, p = 0.00242
- C(musical_experience) chi^2(1, N = 303) = 12.09231, p = 0.00051
- C(condition) chi^2(1, N = 303) = 651.62290, p = 0.00000
- age chi^2(1, N = 303) = 9.33098, p = 0.00225
- preferred_period chi^2(1, N = 303) = 8.66342, p = 0.00325
"""

## Plot Musical Experience
musical_experience_avg = merged_results_df.groupby(["condition","trial_speed","musical_experience"]).agg({"error":[np.mean,std_error]}).reset_index()
bar_width = .95 / 2
fig, ax = plt.subplots(1,2, figsize = standard_fig, sharey = True)
for c, cond in enumerate(["paced","unpaced"]):
    for s, speed in enumerate(["SpedUp","NoChange","SlowedDown"]):
        for e, experience in enumerate([0,1]):
            data_to_plot = musical_experience_avg.loc[(musical_experience_avg.condition == cond)&
                                                        (musical_experience_avg.trial_speed==speed)&
                                                        (musical_experience_avg.musical_experience==experience)]
            ax[c].bar(0.025 + s + bar_width*e, data_to_plot["error"]["mean"],
                    yerr=data_to_plot["error"]["<lambda>"],
                    color = "blueviolet" if e == 0 else "orange", edgecolor = "darkviolet" if e == 0 else "darkorange", align = "edge", width = bar_width,
                    label = "" if s != 0 else "No Musical Experience" if e == 0 else "Has Musical Experience", alpha = .5)
for a in ax:
    a.set_xticks(np.arange(3)+.5)
    a.set_xticklabels(["80%","100%","120%"])
    a.tick_params(labelsize = 14)
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)
ax[0].set_ylabel("Error", fontsize = 16, multialignment = "center", labelpad = 15, fontweight = "bold")
ax[0].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
ax[0].set_title("Synchronization", fontsize = 14); ax[1].set_title("Continuation", fontsize = 14)
ax[0].legend(loc = "upper left", fontsize = 12)
fig.tight_layout()
fig.subplots_adjust(wspace = .1, bottom = .13)
fig.text(0.55, 0.02, 'Metronome Condition', ha='center', fontsize = 14, fontweight = "bold")
fig.savefig(stats_plots + "error_musical_experience.png", dpi=300)
plt.close()

## Plot Age
error_aggs = lambda vals: tuple(bootstrap_ci(vals, sample_percent = 30, samples = 1000))
age_bin_ci = merged_results_df.groupby(["condition","age_bin"]).agg(error_aggs).reset_index()
age_bin_sem = merged_results_df.groupby(["condition","age_bin"]).agg({"error":std_error}).reset_index()
for i in range(3): age_bin_ci[i] = age_bin_ci["error"].map(lambda j: j[i])
fig, ax = plt.subplots(1,1, figsize = standard_fig, sharey = True)
for c, cond in enumerate(["paced","unpaced"]):
    ci_data_to_plot = age_bin_ci.loc[age_bin_ci.condition == cond]
    se_data_to_plot = age_bin_sem.loc[age_bin_sem.condition == cond]
    ax.errorbar(age_bin_points, ci_data_to_plot[1].values,
                        yerr = se_data_to_plot["error"].values,
                        color = "teal" if c == 0 else "slategray", linewidth = 2, alpha = .5)
    ax.fill_between([age_bin_points[0]-1] + age_bin_points[1:-1] + [age_bin_points[-1]+1], ci_data_to_plot[0].values, ci_data_to_plot[2].values,
                    color = "teal" if c == 0 else "slategray", alpha = .3,
                    label = "Synchronization" if c == 0 else "Continuation")
ax.set_ylim(bottom = 0, top = 15)
ax.set_xlabel("Age", fontsize = 16, labelpad = 10, fontweight = "bold")
ax.tick_params(labelsize = 14)
ax.set_xticks(age_bin_points); ax.set_xticklabels(age_bin_strings)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel("Error", fontsize = 16, labelpad = 10, fontweight = "bold")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
ax.legend(loc = "upper right", frameon = True, facecolor = "white", fontsize = 14)
fig.tight_layout()
fig.savefig(stats_plots + "error_age.png", dpi=300)
plt.close()


################################################################################
### Drift Analysis (Directional)
################################################################################

## Fit Mixed LM Model
drift_model_formula = "drift ~ C(gender) + age + preferred_period + trial_speed + C(musical_experience)"
drift_model = smf.mixedlm(drift_model_formula, data = nomet_df, groups = nomet_df["subject"]).fit(reml=True)
drift_summary, drift_wald = summarize_lm_model(drift_model)

print("Drift Effects:"); print_significant(drift_model, drift_wald)

"""
Drift Effects:
- trial_speed chi^2(2, N = 303) = 49.25980, p = 0.00000
"""

## T-Tests against no drift
slowed_t = sm.stats.ttest_ind(np.zeros(int(len(nomet_df)/3)), nomet_df.loc[nomet_df.trial_speed == "SlowedDown"]["drift"].values)
constant_t = sm.stats.ttest_ind(np.zeros(int(len(nomet_df)/3)), nomet_df.loc[nomet_df.trial_speed == "NoChange"]["drift"].values)
sped_t = sm.stats.ttest_ind(np.zeros(int(len(nomet_df)/3)), nomet_df.loc[nomet_df.trial_speed == "SpedUp"]["drift"].values)

"""
- slowed_t (1.3275567940880735, 0.1848260950433708, 604.0)
- sped_t (-6.2771137220428477, 6.5885968506837101e-10, 604.0)
- constant_t (-2.1877377970784138, 0.029071224825391283, 604.0)
"""

## Standard Bar Plot of Drift vs. Trial Speed
drift_by_trial_speed_avg = nomet_df.groupby(["trial_speed"]).agg({"drift":[np.mean, std_error]})
fig, ax = plt.subplots(1, 1, figsize = standard_fig, sharex = True, sharey = True)
for t, trial_speed in enumerate(["SpedUp","NoChange","SlowedDown"]):
    data_to_plot = drift_by_trial_speed_avg.loc[trial_speed]
    ax.bar(t, data_to_plot["drift"]["mean"],
            yerr = data_to_plot["drift"]["<lambda>"],
            color = "blue", alpha = .5, edgecolor = "navy")
ax.axhline(0, color = "black", linewidth = 1)
ax.set_xticks(np.arange(3))
ax.set_xticklabels(["80%\nPreferred Period","100%\nPreferred Period","120%\nPreferred Period"])
ax.set_xlabel("Metronome Condition", fontsize = 16, labelpad = 15, fontweight = "bold")
ax.tick_params(labelsize = 14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel("Drift (ITI Percent Change)", fontsize = 16, fontweight = "bold")
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
ax.set_ylim(-1.2,3.2)
fig.tight_layout()
fig.savefig(stats_plots + "drift_trialspeed_bar.png", dpi=300)
plt.close()

## Standard Bar of Drift vs. Trial Speed, broken down by musicianship
drift_by_trial_speed_me_avg = nomet_df.groupby(["trial_speed","musical_experience"]).agg({"drift": [np.mean, std_error]})
bar_width = .95/2
fig, ax = plt.subplots(1, 1, figsize = standard_fig, sharex = True, sharey = True)
for t, trial_speed in enumerate(["SpedUp","NoChange","SlowedDown"]):
    for m in [0, 1]:
        data_to_plot = drift_by_trial_speed_me_avg.loc[trial_speed, m]
        ax.bar(0.025 + t + m*bar_width, data_to_plot["drift"]["mean"],
                yerr =data_to_plot["drift"]["<lambda>"],
                color = "blueviolet" if m == 0 else "orange", alpha = .5, edgecolor ="darkviolet" if m == 0 else "darkorange",
                label = {0:"No Musical Experience",1:"Has Musical Experience"}[m] if t == 0 else "",
                width = bar_width, align = "edge")
ax.axhline(0, color = "black", linewidth = 1)
ax.set_xticks(np.arange(3)+.5)
ax.set_xticklabels(["80%\nPreferred Period","100%\nPreferred Period","120%\nPreferred Period"])
ax.set_xlabel("Metronome Condition", fontsize = 16, labelpad = 15, fontweight = "bold")
ax.tick_params(labelsize = 14)
ax.set_ylabel("Drift (ITI Percent Change)", fontsize = 16, fontweight = "bold")
ax.legend(loc = "upper right", fontsize = 14, frameon = True, facecolor = "white")
ax.set_ylim(-2.2,3.2)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
fig.tight_layout()
fig.savefig(stats_plots + "drift_trialspeed_musicalexperience_bar.png", dpi=300)
plt.close()

## Standard Bar of Drift vs. Trial Speed, broken down by gender
drift_by_trial_speed_gender_avg = nomet_df.groupby(["trial_speed","gender"]).agg({"drift": [np.mean, std_error]})
bar_width = .95/2
fig, ax = plt.subplots(1, 1, figsize = standard_fig, sharex = True, sharey = True)
for t, trial_speed in enumerate(["SpedUp","NoChange","SlowedDown"]):
    for m in [0, 1]:
        data_to_plot = drift_by_trial_speed_gender_avg.loc[trial_speed, m]
        ax.bar(0.025 + t + m*bar_width, data_to_plot["drift"]["mean"],
                yerr = data_to_plot["drift"]["<lambda>"],
                color = "blue" if m == 0 else "red", alpha = .5, edgecolor = "navy" if m == 0 else "darkred",
                label = {0:"Male",1:"Female"}[m] if t == 0 else "",
                width = bar_width, align = "edge")
ax.axhline(0, color = "black", linewidth = 1)
ax.set_xticks(np.arange(3)+.5)
ax.set_xticklabels(["80%\nPreferred Period","100%\nPreferred Period","120%\nPreferred Period"])
ax.set_xlabel("Metronome Condition", fontsize = 16, labelpad = 15, fontweight = "bold")
ax.tick_params(labelsize = 14)
ax.set_ylabel("Drift (ITI Percent Change)", fontsize = 16, fontweight = "bold")
ax.legend(loc = "upper right", fontsize = 14, frameon = True, facecolor = "white")
ax.set_ylim(-1.9,3.7)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
fig.tight_layout()
fig.savefig(stats_plots + "drift_trialspeed_gender_bar.png", dpi=300)
plt.close()

## Combined Standard Drift + Age Gender
fig, ax = plt.subplots(1, 2, figsize = standard_fig, sharex = False, sharey = True)
for t, trial_speed in enumerate(["SpedUp","NoChange","SlowedDown"]):
    data_to_plot = drift_by_trial_speed_avg.loc[trial_speed]
    ax[0].bar(t, data_to_plot["drift"]["mean"],
            yerr = data_to_plot["drift"]["<lambda>"],
            color = "blue", alpha = .5, edgecolor = "navy")
ax[0].set_xticks(np.arange(3))
for t, trial_speed in enumerate(["SpedUp","NoChange","SlowedDown"]):
    for m in [0, 1]:
        data_to_plot = drift_by_trial_speed_gender_avg.loc[trial_speed, m]
        ax[1].bar(0.025 + t + m*bar_width, data_to_plot["drift"]["mean"],
                yerr = data_to_plot["drift"]["<lambda>"],
                color = "blue" if m == 0 else "red", alpha = .5, edgecolor = "navy" if m == 0 else "darkred",
                label = {0:"Male",1:"Female"}[m] if t == 0 else "",
                width = bar_width, align = "edge")
ax[1].set_xticks(np.arange(3)+.5)
ax[1].legend(loc = "upper right", fontsize = 14, frameon = True, facecolor = "white")
for i in range(2):
    ax[i].axhline(0, color = "black", linewidth = 1)
    ax[i].set_xticklabels(["80%","100%","120%"])
    ax[i].tick_params(labelsize = 14)
    ax[i].set_xlabel("Metronome Condition", fontsize = 16, labelpad = 15, fontweight = "bold")
    ax[i].spines['right'].set_visible(False)
    ax[i].spines['top'].set_visible(False)
    ax[i].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
ax[0].set_ylabel("Drift (ITI Percent Change)", fontsize = 16, fontweight = "bold")
fig.tight_layout()
fig.savefig(stats_plots + "combined_drift_standard_and_gender.png", dpi=300)
plt.close()

## Within-Subject
drift_pivot = pd.merge(pd.pivot_table(nomet_df, index = ["subject"],columns=["trial_speed"], values = ["drift"]), nomet_df.drop_duplicates("subject")[merge_vars],left_index=True,right_on="subject")
drift_pivot["slowed_change"] = drift_pivot[("drift","SlowedDown")]-drift_pivot[("drift","NoChange")] # Want this negative
drift_pivot["sped_change"] = drift_pivot[("drift","SpedUp")]-drift_pivot[("drift","NoChange")] # Want this positive
drift_pivot_melted = pd.melt(drift_pivot, id_vars = merge_vars, value_vars = ["slowed_change","sped_change"]).sort_values(["subject","variable"])

## T-Tests
slowed_t = sm.stats.ttest_ind(np.zeros(int(len(drift_pivot_melted)/2)), drift_pivot_melted.loc[drift_pivot_melted.variable == "slowed_change"]["value"].values)
sped_t = sm.stats.ttest_ind(np.zeros(int(len(drift_pivot_melted)/2)), drift_pivot_melted.loc[drift_pivot_melted.variable == "sped_change"]["value"].values)

"""
Slowed condition -> (3.7655544977830351, 0.00018242710331818079, 604.0)
Sped condition -> (-3.7575886298128562, 0.00018819354900583791, 604.0)
"""

## Mixed LM Model
rel_model_test_form = "value ~ C(gender) + age + preferred_period + variable + C(musical_experience)"
rel_model = smf.mixedlm(rel_model_test_form, data = drift_pivot_melted, groups = drift_pivot_melted["subject"]).fit()
rel_summary, rel_wald = summarize_lm_model(rel_model)

print("Within-Subject Drift Effects:"); print_significant(rel_model, rel_wald)

"""
Within-Subject Drift Effects:
- variable chi^2(1, N = 303) = 39.12849, p = 0.00000
"""

## Plot Within-Subject Drift
rel_drift_by_cond = drift_pivot_melted.groupby(["variable"]).agg({"value":[np.mean, std_error]}).reset_index()
fig, ax = plt.subplots(figsize = standard_fig)
for j, var in enumerate(["sped_change","slowed_change"]):
    data_to_plot = rel_drift_by_cond.loc[rel_drift_by_cond.variable == var]
    ax.bar(j, data_to_plot["value"]["mean"],
            yerr=data_to_plot["value"]["<lambda>"],
            color = "blue", edgecolor = "navy", alpha = .5)
ax.axhline(0, color = "black")
ax.set_xticks([0,1]); ax.set_xticklabels(["80%\nPreferred Period","120%\nPreferred Period"])
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel("Metronome Condition", fontsize = 14, labelpad = 20, fontweight = "bold")
ax.set_ylabel("Difference from 100%\nPreferred Period", ha = "center", va = "center", multialignment="center", labelpad = 20,
                fontsize = 14, fontweight = "bold")
ax.tick_params(labelsize = 14)
fig.tight_layout()
fig.savefig(stats_plots + "within_subject_drift.png", dpi=300)
plt.close()

################################################################################
### Variability Analysis
################################################################################

## Fit Mixed LM Model
cv_model_form = "cv ~ C(gender) + age + preferred_period + trial_speed + C(musical_experience) + C(condition)"
cv_model = smf.mixedlm(cv_model_form, data = merged_results_df, groups = merged_results_df["subject"]).fit(reml=True)
cv_model_summary, cv_wald = summarize_lm_model(cv_model)

print("Variability Effects:"); print_significant(cv_model, cv_wald)

"""
Variability Effects:
- trial_speed chi^2(2, N = 303) = 28.25395, p = 0.00000
- C(musical_experience) chi^2(1, N = 303) = 14.03075, p = 0.00018
- C(condition) chi^2(1, N = 303) = 47.48795, p = 0.00000
- age chi^2(1, N = 303) = 63.15085, p = 0.00000
- preferred_period chi^2(1, N = 303) = 10.08424, p = 0.00150
"""

## Age Effects
cv_aggs = {"cv": lambda values: tuple(bootstrap_ci(values, sample_percent = 30))}
age_bin_var_sem = merged_results_df.groupby(["condition","age_bin"]).agg({"cv":std_error}).reset_index()
age_bin_var_ci = merged_results_df.groupby(["condition","age_bin"]).agg(cv_aggs).reset_index()
for i in range(3): age_bin_var_ci[i] = age_bin_var_ci["cv"].map(lambda j: j[i])
fig, ax = plt.subplots(1, 1, figsize = standard_fig)
for c, cond in enumerate(["paced","unpaced"]):
    avg_data_to_plot = age_bin_var_sem.loc[age_bin_var_sem.condition == cond]
    ci_data_to_plot = age_bin_var_ci.loc[age_bin_var_ci.condition == cond]
    ax.errorbar(age_bin_points, ci_data_to_plot[1].values,
                        yerr = avg_data_to_plot["cv"].values,
                        color = "teal" if c == 0 else "slategray", linewidth = 2, alpha = .7)
    ax.fill_between([age_bin_points[0]-1] + age_bin_points[1:-1] + [age_bin_points[-1]+1], ci_data_to_plot[0].values, ci_data_to_plot[2].values,
                    color = "teal" if c == 0 else "slategray", alpha = .3,
                    label = "Synchronization" if c == 0 else "Continuation")
ax.set_xlabel("Age", fontsize = 16, labelpad = 10, fontweight = "bold")
ax.set_xticks(age_bin_points); ax.set_xticklabels(age_bin_strings)
ax.tick_params(labelsize = 14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_ylabel("Coefficient of Variation", fontsize = 16, labelpad = 10, fontweight = "bold")
ax.legend(loc = "upper right", frameon = True, facecolor = "white", fontsize = 14)
fig.tight_layout()
fig.savefig(stats_plots + "variability_age.png", dpi=300)
plt.close()


## Trial Speed
speed_var_avg = merged_results_df.groupby(["condition","trial_speed"]).agg({"cv":[np.mean, std_error]}).reset_index()
bar_width = .95/2
fig, ax = plt.subplots(figsize = standard_fig)
for c, cond in enumerate(["paced","unpaced"]):
    for t, trial_speed in enumerate(["SpedUp","NoChange","SlowedDown"]):
        data_to_plot = speed_var_avg.loc[(speed_var_avg.condition==cond)&(speed_var_avg.trial_speed==trial_speed)]
        ax.bar(0.025 + t + bar_width*c, data_to_plot["cv"]["mean"],
                yerr = data_to_plot["cv"]["<lambda>"],
                align = "edge", width = bar_width, color = "teal" if c == 0 else "slategray",
                edgecolor = "darkcyan" if c == 0 else "black",
                label = {0:"Synchronization",1:"Continuation"}[c] if t == 0 else "",
                alpha = .5)
ax.set_xticks(np.arange(3)+.5)
ax.set_xticklabels(["80%\nPreferred Period","100%\nPreferred Period","120%\nPreferred Period"])
ax.set_xlabel("Metronome Condition", fontsize = 16, labelpad = 15, fontweight = "bold")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.tick_params(labelsize = 14)
ax.set_ylabel("Coefficient of Variation", fontsize = 16, labelpad = 15, fontweight = "bold")
ax.legend(loc = "upper right", frameon = True, facecolor = "white", fontsize = 12)
ax.set_ylim(top = 0.07)
fig.tight_layout()
fig.savefig(stats_plots + "variability_trialspeed.png", dpi=300)
plt.close()

## Musical Experience + Variability
cv_music_avg = merged_results_df.groupby(["condition","trial_speed","musical_experience"]).agg({"cv":[np.mean, std_error]}).reset_index()
fig, ax = plt.subplots(1, 2, figsize = standard_fig, sharey = True, sharex = True)
for c, cond in enumerate(["paced","unpaced"]):
    for me in range(2):
        data_to_plot = cv_music_avg.set_index(["condition","musical_experience"]).loc[cond, me]
        data_to_plot = data_to_plot.set_index("trial_speed").loc[["SpedUp","NoChange","SlowedDown"]]
        ax[c].bar(0.025 + np.arange(3) + .95/2*me, data_to_plot["cv"]["mean"].values,
                    yerr = data_to_plot["cv"]["<lambda>"].values,
                    color = "blueviolet" if me == 0 else "orange", edgecolor = "darkviolet" if me == 0 else "darkorange",
                    width = .95/2, alpha = .5, align = "edge",
                    label = "Has Musical Experience" if me == 1 else "No Musical Experience")
        ax[c].spines['right'].set_visible(False)
        ax[c].spines['top'].set_visible(False)
        ax[c].set_xticks(np.arange(3) + .5)
        ax[c].tick_params(labelsize = 14)
        ax[c].set_xticklabels(["80%","100%","120%"])
ax[0].set_title("Synchronization", fontsize = 16); ax[1].set_title("Continuation", fontsize = 16)
ax[0].set_ylabel("Coefficient of\nVariation", fontsize = 16, labelpad = 15, fontweight = "bold")
ax[0].legend(loc = "upper right", fontsize = 12, frameon = True, facecolor = "white")
fig.text(0.55, 0.02, 'Metronome Condition', ha='center', fontsize = 14, fontweight = "bold")
fig.tight_layout()
fig.subplots_adjust(top = .9, bottom = .13)
fig.savefig(stats_plots + "variability_musicalexperience_trialspeed.png", dpi=300)
plt.close()

################################################################################
### Subject Filtering
################################################################################

## Want to understand which subjects were thrown out (and why)

## Read in Survey Data
survey_data_full = pd.read_csv("./data/survey.csv")
survey_data_full["Subject"] = survey_data_full.Subject.map(int)

################
### Stage 1
################

## Load Results
stage_1_results = load_pickle("./data/manual_inspection.pickle")

## Format Results
stage_1_results = pd.Series(stage_1_results).reset_index().rename(columns = {"index":"file",0:"pass"})
stage_1_results["subject"] = stage_1_results["file"].map(lambda i: i.split("/")[-1].replace(".mat","")).map(int)

## Extract Non-Passing + Merge Demos
stage_1_nonpassing = stage_1_results.loc[stage_1_results["pass"] != "pass"].copy()
stage_1_nonpassing = pd.merge(stage_1_nonpassing, survey_data_full[["Subject","Age","Gender"]], left_on = "subject", right_on = "Subject", how = "left")

"""
12 subjects removed due to sensor issue
3 subjects removed due to forgetting to tap (2x10yrs, 1x5yrs)
2 subject removed due to tapping style (1x6yrs, 1x14yrs)
1 subject removed due to missing survey data
"""

s1_filtered_subjects = stage_1_nonpassing.subject.unique()

################
### Stage 2
################

## Load Results
stage_2_results = load_pickle("./data/stage_2_processed.pickle")

## Identify Failures
stage_2_failures = pd.DataFrame(stage_2_results).T.isnull().sum(axis=1).reset_index().rename(columns = {"index":"subject",0:"failures"})
stage_2_failures_nonzero = stage_2_failures.loc[stage_2_failures.failures>0].copy()
stage_2_failures_nonzero["subject"] = stage_2_failures_nonzero["subject"].map(int)

## Identify Subject not filtered previously
stage_2_failures_nonzero_new = stage_2_failures_nonzero.loc[~stage_2_failures_nonzero.subject.isin(stage_1_nonpassing.subject)]

"""
All subjects were successfully processed
"""

################
### Stage 3
################

## Load Results
stage_3_results = load_pickle("./data/stage_3_processed.pickle")

## Identify Filtering
filtered_df = []
for subject, subject_res in stage_3_results.items():
    if subject_res is None:
        for i in range(1,7):
            filtered_df.append([subject, i, True, "previous"])
        continue
    for trial, trial_res in subject_res.items():
        filtered_df.append([subject, trial, trial_res["filtered"],trial_res["discard_reason"]])
filtered_df = pd.DataFrame(filtered_df,columns = ["subject","trial","filtered","discard_reason"])

## Locate Non-Prior Filtered Subjects
filtered_df_nonprior = filtered_df.loc[(filtered_df.filtered) & ~(filtered_df.discard_reason.isin(["previous",None]))].copy()
filtered_df_nonprior = pd.merge(filtered_df_nonprior, survey_data_full[["Subject","Age","Gender"]], left_on = "subject", right_on = "Subject", how = "left")

"""
6 subjects experienced some filtering
- Too Much Noise (2 subjects - 1x6yrs and 1x25yrs)
- Too Light (3 subjects - 1x27yrs, 2x7rs)
- Too Much Variation (1 subject - 1x5yrs)
"""

s3_filtered_subjects = filtered_df_nonprior.subject.unique()

################
### Metric-based Exclusion
################

## Identify ages of other subjects filtered due to metric computation issues
previously_filtered = list(s1_filtered_subjects) + list(s3_filtered_subjects)

## Additional Filtered
additional_filtered = survey_data_full.loc[~(survey_data_full.Subject.isin(previously_filtered)) & ~(survey_data_full.Subject.isin(subject_deduped.subject)) &
                                            (survey_data_full.Subject.isin(stage_1_results.subject))]

"""
5 subjects were filtered because the experimental protocol was set up improperly (wrong trial length)
- Ages (32, 58, 55, 12, 44)

3 Subjects were filtered for not having enough taps during the specified period to get a proper metric calculation
- Ages (10, 26, 18)
"""
