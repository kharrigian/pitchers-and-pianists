
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

###############
### Plot Helpers
###############

## Plotting Variables
standard_fig = (10,5.8)
plot_dir = "./plots/"
stats_plots = plot_dir + "analysis/"

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

## Bootstrapped Confidence Interval
def bootstrap_ci(values, alpha = 0.05, func = np.mean, sample_percent = 20, samples = 1000):
    processed_vals = []
    values = np.array(values)
    for sample in range(samples):
        sample_vals = np.random.choice(values, int(values.shape[0] * sample_percent/100.), replace = False)
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
cv_cont_metrics = ["nomet_cv_first5","nomet_cv_last5"]
cv_metrics = cv_sync_metrics + cv_cont_metrics

# Error Variables
error_sync_metrics = ["met_sync_error_last5", "met_sync_error_last5_rel"]
error_cont_metrics = ["nomet_sync_error_first5","nomet_sync_error_last5","nomet_sync_error_first5_rel","nomet_sync_error_last5_rel"]
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
age_bins = [5,10,20,30,40,50]
age_bin_strings = create_bin_strings(results.age, age_bins)
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
for error in ["met_sync_error_last5","nomet_sync_error_first5","nomet_sync_error_last5","nomet_drift"]:
    results["abs_%s" % error] = np.abs(results[error])
    results["abs_%s_rel" % error] = np.abs(results[error+"_rel"])

# Absolute Drift in Regression Coefficient
results["abs_nomet_drift_regression"] = np.abs(results["nomet_drift_regression"])

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
fig, ax = plt.subplots(1,1, figsize = standard_fig, sharey = False, sharex = False)
ax.hist(subject_deduped["preferred_period"], bins = 30, normed = False, color = "lightgray", edgecolor = "black")
ax.set_xlabel("Preferred Period (ms)", fontsize = 18)
ax.set_ylabel("Subjects", fontsize = 18)
ax.tick_params(labelsize=16)
fig.tight_layout()
plt.savefig(stats_plots + "preferred_period")
plt.close()

## No Correlation between age and preferred period
fig, ax = plt.subplots(1,1, figsize = standard_fig, sharey = False, sharex = False)
ax.scatter(subject_deduped["age"], subject_deduped["preferred_period"], color = "lightgray", edgecolor="black", alpha = 1,
            s = 50)
ax.set_xlabel("Age", fontsize = 18)
ax.set_ylabel("Preferred period (ms)", fontsize = 18)
ax.tick_params(labelsize=16)
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
        ax.bar(0.025 + g*0.45 + a, demo_count, color = {"Male":"lightgray","Female":"white"}[gender],
        alpha = 1, align = "edge", width = .45, label = gender if a == 0 else "", edgecolor = "black")
        ax.text(0.25 + g*0.45 + a, demo_count + 1, demo_count, ha = "center", fontsize = 18)
        max_count = demo_count if demo_count > max_count else max_count
ax.legend(loc = "upper right", frameon = True, fontsize = 18)
ax.set_xticks(np.arange(a+1)+.5)
ticks = ax.set_xticklabels(age_bin_strings, rotation = 45)
ax.set_xlabel("Age", fontsize = 18)
ax.set_ylabel("Subjects", fontsize = 18)
lim = ax.set_ylim(0, max_count+5)
ax.tick_params(labelsize = 16)
fig.tight_layout()
fig.savefig(stats_plots + "demographics.png")
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
musical_experience_dist.plot.barh(ax = ax[0], color = "lightgray", edgecolor = "black")
ax[0].set_yticks(np.arange(3)); ax[0].set_yticklabels(musical_experience_dist.index.values, multialignment = "center")
ax[1].scatter(music_exp_subset["age"], music_exp_subset["musical_experience_yrs"], color = "lightgray", edgecolor = "black", s = 50)
ax[1].plot([0, music_exp_subset.age.max()], [0, music_exp_subset.age.max()], linestyle = "--", color = "black")
ax[0].set_xlabel("Subjects",fontsize=18)
ax[1].set_xlabel("Age",fontsize=18); ax[1].set_ylabel("Years of Musical Experience",fontsize=18)
for a in ax:
    a.tick_params(labelsize = 16)
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
        ax[0].bar(0.025 + x*.45 + g, demo_counts, color = {0:"white",1:"lightgray"}[experience],
                  alpha = 1, label = {1:"Has Musical Experience",0:"No Musical Experience"}[experience] if g == 0 else "",
                  width = .45, align = "edge", edgecolor = "black")
        ax[0].text(0.025+ x*.45 + .45/2 + g, demo_counts + 4, demo_counts + 1, ha="center", fontsize = 16)
        max_gender_count = demo_counts if demo_counts > max_gender_count else max_gender_count

    for a, age in enumerate(subject_deduped.age_bin.unique()):
        demo_counts = len(subject_deduped.loc[(subject_deduped.musical_experience==experience)&
                                             (subject_deduped.age_bin==age)])
        ax[1].bar(0.025 + x*.45 + a, demo_counts, color = {0:"white",1:"lightgray"}[experience],
                  alpha = 1, label =  {1:"Has Musical Experience",0:"No Musical Experience"}[experience] if a == 0 else "",
                  width = .45, align = "edge", edgecolor = "black")
        ax[1].text(0.025+ x*.45 +.45/2 + a, demo_counts + 1, demo_counts + 1, ha="center", fontsize = 16)
        max_age_count = demo_counts if demo_counts > max_age_count else max_age_count
ax[0].set_ylim(0, max_gender_count+20)
ax[1].set_ylim(0, max_age_count + 20)
ax[0].set_xticks(np.arange(2)+.5)
ax[0].set_xticklabels(["Female","Male"])
ax[1].set_xticks(np.arange(a+1)+.5)
ax[1].set_xticklabels(age_bin_strings, rotation = 45)
handles, labels = ax[0].get_legend_handles_labels()
leg = fig.legend(handles, labels, loc='upper center', ncol = 2, fontsize = 16)
for t in leg.texts:
    t.set_multialignment('center')
for a in ax:
    a.set_ylabel("Subjects", fontsize = 18)
    a.tick_params(labelsize = 16)
fig.tight_layout()
fig.subplots_adjust(top = .86)
plt.savefig(stats_plots + "musical_experience_demographics")
plt.close()

################################################################################
### Synchronization Error Analysis
################################################################################

## Choose Metrics
sync_error = "abs_met_sync_error_last5_rel"

## Fit Mixed LM Model
sync_error_model_formula = "{} ~ C(gender) + age + preferred_period + trial_speed + C(musical_experience) + C(healthy)".format(sync_error)
sync_error_model = smf.mixedlm(sync_error_model_formula, data = mean_results, groups = mean_results["subject"]).fit(reml=True)
sync_error_summary, sync_error_wald = summarize_lm_model(sync_error_model)

"""
Musical Experience, Trial Speed, Age, and Preferred Period have significant effects
"""

## Musical Experience
sync_error_musical_experience_avg = mean_results.groupby(["trial_speed","musical_experience"]).agg({sync_error:
                                                    lambda values: tuple(bootstrap_ci(values, sample_percent = 75))}).reset_index()
for i in range(3): sync_error_musical_experience_avg[i] = sync_error_musical_experience_avg[sync_error].map(lambda j: j[i])
bar_width = .95/2
fig, ax = plt.subplots(1,1, figsize = standard_fig, sharey = True)
for t, trial_speed in enumerate(["SlowedDown","NoChange","SpedUp"]):
    for m, me in enumerate([0,1]):
        data_to_plot = sync_error_musical_experience_avg.set_index(["trial_speed","musical_experience"]).loc[trial_speed, me]
        ax.bar(0.025 + t + m*bar_width, data_to_plot[1], yerr = np.array([data_to_plot[1]-data_to_plot[0], data_to_plot[2]-data_to_plot[1]]).reshape(-1,1),
                    color = {0:"white",1:"lightgray"}[m], edgecolor = "black", align = "edge", width = bar_width,
                    label = {0:"No Musical Experience",1:"Has Musical Experience"}[m] if t == 0 else "")
ax.legend(loc = "upper left", frameon = True, facecolor = "white", fontsize = 18)
ax.set_xticks(np.arange(3)+.5); ax.set_xticklabels(["Slowed Down","No Change","Sped Up"])
ax.set_xlabel("Trial Condition", fontsize = 16, labelpad = 15)
ax.tick_params(labelsize = 14)
ax.set_ylabel("Synchronization Error\n(Relative to Trial Period)", fontsize = 16, multialignment = "center", labelpad = 15)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
fig.tight_layout()
fig.savefig(stats_plots + "sync_error_musical_experience_trial_speed.png")
plt.close()

## Trial Speed
sync_error_trial_speed = mean_results.groupby(["trial_speed"]).agg({sync_error:
                                                    lambda values: tuple(bootstrap_ci(values, sample_percent = 75))}).reset_index()
for i in range(3): sync_error_trial_speed[i] = sync_error_trial_speed[sync_error].map(lambda j: j[i])
fig, ax = plt.subplots(1, 1, figsize = standard_fig)
for t, trial_speed in enumerate(["SlowedDown","NoChange","SpedUp"]):
    data_to_plot = sync_error_trial_speed.set_index(["trial_speed"]).loc[trial_speed]
    ax.bar(0.025 + t, data_to_plot[1], yerr = np.array([data_to_plot[1]-data_to_plot[0], data_to_plot[2]-data_to_plot[1]]).reshape(-1,1),
                    color = "lightgray", edgecolor = "black", align = "edge", width = .95)
ax.set_xticks(np.arange(3)+.5); ax.set_xticklabels(["Slowed Down","No Change","Sped Up"])
ax.set_xlabel("Trial Condition", fontsize = 16, labelpad = 15)
ax.tick_params(labelsize = 14)
ax.set_ylabel("Synchronization Error\n(Relative to Trial Period)", fontsize = 16, multialignment = "center", labelpad = 15)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
fig.tight_layout()
fig.savefig(stats_plots + "sync_error_trial_speed.png")
plt.close()

## Preferred Period
fig, ax = plt.subplots(1,3,figsize = standard_fig, sharey = True)
for t, trial_speed in enumerate(["SlowedDown","NoChange","SpedUp"]):
    data_to_plot = mean_results.loc[mean_results.trial_speed == trial_speed]
    ax[t].scatter(data_to_plot["preferred_period"], data_to_plot[sync_error], s = 50, color = "lightgray",
                    edgecolor="black", alpha = .75)
    ax[t].set_xlabel("Preferred Period (ms)", fontsize = 16)
    ax[t].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
    ax[t].set_title({"NoChange":"No Change","SlowedDown":"Slowed Down","SpedUp":"Sped Up"}[trial_speed], fontsize = 16)
    ax[t].tick_params(labelsize = 14)
ax[0].set_ylabel("Synchronization Error\n(Relative to Trial Period)", fontsize = 16, multialignment = "center", labelpad = 15)
fig.tight_layout()
fig.savefig(stats_plots + "sync_error_preferred_period.png")
plt.close()

## Age
sync_error_age_bin =  mean_results.groupby(["trial_speed","age_bin"]).agg({sync_error:
                                                    lambda values: tuple(bootstrap_ci(values, sample_percent = 75))}).reset_index()
for i in range(3): sync_error_age_bin[i] = sync_error_age_bin[sync_error].map(lambda j: j[i])
age_bin_points = [7.5, 15, 25, 35, 45, 59]
fig, ax = plt.subplots(1, 3, figsize = standard_fig, sharey = True)
for t, trial_speed in enumerate(["SlowedDown","NoChange","SpedUp"]):
    data_to_plot = mean_results.loc[mean_results.trial_speed == trial_speed]
    age_bin_data_to_plot = sync_error_age_bin.set_index("trial_speed").loc[trial_speed].sort_values("age_bin")
    ax[t].scatter(data_to_plot["age"], data_to_plot[sync_error], s = 50, color = "lightgray",
                    edgecolor="black", alpha = .3)
    ax[t].errorbar(age_bin_points, age_bin_data_to_plot[1].values, yerr = np.array([age_bin_data_to_plot[1]-age_bin_data_to_plot[0], age_bin_data_to_plot[2]-age_bin_data_to_plot[1]]),
                color = "black", linewidth = 3)
    ax[t].set_ylim(0,15)
    ax[t].tick_params(labelsize = 14)
    ax[t].set_xlabel("Age", fontsize = 16)
    ax[t].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
    ax[t].set_title({"NoChange":"No Change","SlowedDown":"Slowed Down","SpedUp":"Sped Up"}[trial_speed], fontsize = 16)
ax[t].tick_params(labelsize = 14)
ax[0].set_ylabel("Synchronization Error\n(Relative to Trial Period)", fontsize = 16, multialignment = "center", labelpad = 15)
fig.tight_layout()
fig.savefig(stats_plots + "sync_error_age.png")
plt.close()

################################################################################
### Continuation Error Analysis
################################################################################

## Choose continuation error metric
cont_error = "abs_nomet_sync_error_last5_rel"

## Difference between Synchronization and Continuation Error
sync_cont_avgs = mean_results.groupby(["trial_speed"]).agg({sync_error: lambda values: tuple(bootstrap_ci(values, sample_percent = 75)),
                                                            cont_error: lambda values: tuple(bootstrap_ci(values, sample_percent = 75))})
for col in [sync_error,cont_error]:
    for i in range(3):
        sync_cont_avgs["{}_{}".format(col,i)] = sync_cont_avgs[col].map(lambda j: j[i])
bar_width = .95/2
fig, ax = plt.subplots(1, 1, figsize = standard_fig)
for t, trial_speed in enumerate(["SlowedDown","NoChange","SpedUp"]):
    data_to_plot = sync_cont_avgs.loc[trial_speed]
    for c, col in enumerate([sync_error,cont_error]):
        ax.bar(0.025 + t + c*bar_width, data_to_plot["{}_1".format(col)], yerr = np.array([[data_to_plot["{}_1".format(col)] - data_to_plot["{}_0".format(col)]],
                                                                [data_to_plot["{}_2".format(col)] - data_to_plot["{}_1".format(col)]]]).reshape(-1,1),
                color = {sync_error:"white",cont_error:"lightgray"}[col], edgecolor = "black",
                label = {0:"Synchronization",1:"Continuation"}[c] if t == 0 else "", align = "edge", width = bar_width)
ax.set_xticks(np.arange(3)+0.5); ax.set_xticklabels(["Slowed Down", "No Change", "Sped Up"])
ax.set_xlabel("Trial Condition", fontsize = 16, labelpad = 15)
ax.tick_params(labelsize = 14)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
ax.set_ylabel("Error (Relative to Trial Period)", fontsize = 16)
handles, labels = ax.get_legend_handles_labels()
leg = fig.legend(handles, labels, loc='upper center', ncol = 2, fontsize = 14)
fig.tight_layout()
fig.subplots_adjust(top = .9)
plt.savefig(stats_plots + "sync_error_cont_error_comparison.png")
plt.close()

## Look at correlation between synchronization and continuation error
fig, ax = plt.subplots(1, 3, figsize = standard_fig, sharey = False, sharex = False)
for t, trial_speed in enumerate(["SlowedDown","NoChange","SpedUp"]):
    data_to_plot = mean_results.loc[mean_results.trial_speed == trial_speed].copy()
    ax[t].scatter(data_to_plot[sync_error], data_to_plot[cont_error], s = 50, color = "lightgray",
                    edgecolor="black", alpha = 1)
    ax[t].tick_params(labelsize = 14)
    ax[t].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
    ax[t].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
    ax[t].set_title({"NoChange":"No Change","SlowedDown":"Slowed Down","SpedUp":"Sped Up"}[trial_speed], fontsize = 16)
    corr_model = smf.mixedlm("{} ~ {}".format(cont_error, sync_error), data = data_to_plot, groups = data_to_plot["subject"]).fit()
    data_to_plot["pred"] = corr_model.predict(data_to_plot)
    ax[t].plot(data_to_plot.sort_values(sync_error)[sync_error].values,data_to_plot.sort_values(sync_error)["pred"],
                color = "black", linestyle = ":", linewidth = 2)
    ax[t].set_xlabel("Synchronization Error\n(Relative to Trial Period)", fontsize = 16, multialignment = "center", labelpad = 15)
ax[0].set_ylabel("Continuation Error\n(Relative to Trial Period)", fontsize = 16, multialignment = "center", labelpad = 15)
fig.tight_layout()
plt.close()

## Fit Mixed LM Model
cont_error_model_formula = "{} ~ C(gender) + age + preferred_period + trial_speed + C(musical_experience) + C(healthy)".format(cont_error)
cont_error_model = smf.mixedlm(cont_error_model_formula, data = mean_results, groups = mean_results["subject"]).fit(reml=True)
cont_error_summary, cont_error_wald = summarize_lm_model(cont_error_model)

"""
No factors are significant when accounting for multiplicity. However, musical experience and age both have p_values < 0.05
"""

## Musical Experience
cont_error_musical_experience_avg = mean_results.groupby(["trial_speed","musical_experience"]).agg({cont_error:
                                                    lambda values: tuple(bootstrap_ci(values, sample_percent = 75))}).reset_index()
for i in range(3): cont_error_musical_experience_avg[i] = cont_error_musical_experience_avg[cont_error].map(lambda j: j[i])
bar_width = .95/2
fig, ax = plt.subplots(1,1, figsize = standard_fig, sharey = True)
for t, trial_speed in enumerate(["SlowedDown","NoChange","SpedUp"]):
    for m, me in enumerate([0,1]):
        data_to_plot = cont_error_musical_experience_avg.set_index(["trial_speed","musical_experience"]).loc[trial_speed, me]
        ax.bar(0.025 + t + m*bar_width, data_to_plot[1], yerr = np.array([data_to_plot[1]-data_to_plot[0], data_to_plot[2]-data_to_plot[1]]).reshape(-1,1),
                    color = {0:"white",1:"lightgray"}[m], edgecolor = "black", align = "edge", width = bar_width,
                    label = {0:"No Musical Experience",1:"Has Musical Experience"}[m] if t == 0 else "")
ax.legend(loc = "upper left", frameon = True, facecolor = "white", fontsize = 18)
ax.set_xticks(np.arange(3)+.5); ax.set_xticklabels(["Slowed Down","No Change","Sped Up"])
ax.set_xlabel("Trial Condition", fontsize = 16, labelpad = 15)
ax.tick_params(labelsize = 14)
ax.set_ylim(0, 11)
ax.set_ylabel("Continuation Error\n(Relative to Trial Period)", fontsize = 16, multialignment = "center", labelpad = 15)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
fig.tight_layout()
fig.savefig(stats_plots + "continuation_error_musical_experience_trial_speed.png")
plt.close()

## Age
cont_error_age_bin =  mean_results.groupby(["trial_speed","age_bin"]).agg({cont_error:
                                                    lambda values: tuple(bootstrap_ci(values, sample_percent = 75))}).reset_index()
for i in range(3): cont_error_age_bin[i] = cont_error_age_bin[cont_error].map(lambda j: j[i])
age_bin_points = [7.5, 15, 25, 35, 45, 59]
fig, ax = plt.subplots(1, 3, figsize = standard_fig, sharey = True)
for t, trial_speed in enumerate(["SlowedDown","NoChange","SpedUp"]):
    data_to_plot = mean_results.loc[mean_results.trial_speed == trial_speed]
    age_bin_data_to_plot = cont_error_age_bin.set_index("trial_speed").loc[trial_speed].sort_values("age_bin")
    ax[t].scatter(data_to_plot["age"], data_to_plot[cont_error], s = 50, color = "lightgray",
                    edgecolor="black", alpha = .3)
    ax[t].errorbar(age_bin_points, age_bin_data_to_plot[1].values, yerr = np.array([age_bin_data_to_plot[1]-age_bin_data_to_plot[0], age_bin_data_to_plot[2]-age_bin_data_to_plot[1]]),
                color = "black", linewidth = 3)
    ax[t].set_ylim(0,30)
    ax[t].tick_params(labelsize = 14)
    ax[t].set_xlabel("Age", fontsize = 16)
    ax[t].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
    ax[t].set_title({"NoChange":"No Change","SlowedDown":"Slowed Down","SpedUp":"Sped Up"}[trial_speed], fontsize = 16)
ax[t].tick_params(labelsize = 14)
ax[0].set_ylabel("Continuation Error\n(Relative to Trial Period)", fontsize = 16, multialignment = "center", labelpad = 15)
fig.tight_layout()
fig.savefig(stats_plots + "continuation_error_age.png")
plt.close()

################################################################################
### Drift Analysis (Directional)
################################################################################

## Choose Drift Metric
drift_metric = "nomet_drift_rel"

## Fit Mixed LM Model
drift_model_formula = "{} ~ C(gender) + age + preferred_period + trial_speed + C(musical_experience) + C(healthy)".format(drift_metric)
drift_model = smf.mixedlm(drift_model_formula, data = mean_results, groups = mean_results["subject"]).fit(reml=True)
drift_summary, drift_wald = summarize_lm_model(drift_model)

"""
Trial Speed is the only significant effect
"""

## Histogram of Drift vs. Trial Speed
drift_bins = np.linspace(mean_results[drift_metric].min(), mean_results[drift_metric].max(), 25)
fig, ax = plt.subplots(3, 1, figsize = standard_fig, sharex = True, sharey = True)
for t, trial_speed in enumerate(["SlowedDown","NoChange","SpedUp"]):
    data_to_plot = mean_results.loc[mean_results.trial_speed == trial_speed].copy()
    ax[t].hist(data_to_plot[drift_metric], bins = drift_bins, alpha = 1,
            color = "lightgray", edgecolor = "black")
    ax[t].set_ylabel("Subjects", fontsize = 16)
    ax[t].tick_params(labelsize = 14)
    ax[t].set_title({"NoChange":"No Change","SlowedDown":"Slowed Down","SpedUp":"Sped Up"}[trial_speed], fontsize = 16)
ax[t].set_xlabel("Drift", fontsize = 16)
ax[t].xaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
fig.tight_layout()
fig.subplots_adjust(wspace = .2)
fig.savefig(stats_plots + "drift_trialspeed_histogram.png")
plt.close()

## Standard Bar Plot of Drift vs. Trial Speed
drift_by_trial_speed_avg = mean_results.groupby(["trial_speed"]).agg({drift_metric: lambda values: tuple(bootstrap_ci(values, sample_percent = 75))})
for i in range(3): drift_by_trial_speed_avg[i] = drift_by_trial_speed_avg[drift_metric].map(lambda j: j[i])
fig, ax = plt.subplots(1, 1, figsize = standard_fig, sharex = True, sharey = True)
for t, trial_speed in enumerate(["SlowedDown","NoChange","SpedUp"]):
    data_to_plot = drift_by_trial_speed_avg.loc[trial_speed]
    ax.bar(t, data_to_plot[1],
            yerr = np.array([data_to_plot[1]-data_to_plot[0], data_to_plot[2]-data_to_plot[1]]).reshape(-1,1),
            color = "lightgray", alpha = 1, edgecolor = "black")
ax.axhline(0, color = "black", linewidth = 1)
ax.set_xticks(np.arange(3))
ax.set_xticklabels(["Slowed Down", "No Change", "Sped Up"])
ax.tick_params(labelsize = 14)
ax.set_ylabel("Drift (ITI Percent Change)", fontsize = 16)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
ax.set_ylim(-1.2,1.2)
fig.tight_layout()
fig.savefig(stats_plots + "drift_trialspeed_bar.png")
plt.close()

################################################################################
### Drift Analysis (Absolute)
################################################################################

## Choose Drift Metric
abs_drift_metric = "abs_nomet_drift_rel"

## Fit Mixed LM Model
abs_drift_model_formula = "{} ~ C(gender) + age + preferred_period + trial_speed + C(musical_experience) + C(healthy)".format(abs_drift_metric)
abs_drift_model = smf.mixedlm(abs_drift_model_formula, data = mean_results, groups = mean_results["subject"]).fit(reml=True)
abs_drift_summary, abs_drift_wald = summarize_lm_model(abs_drift_model)

"""
Age has a signficant effect on absolute drift. Musical experience also has an effect, but not significant after correction
"""

## Age Effect (Speed Separation)
abs_drift_trialspeed_avg =  mean_results.groupby(["trial_speed","age_bin"]).agg({abs_drift_metric:
                                                    lambda values: tuple(bootstrap_ci(values, sample_percent = 75))}).reset_index()
for i in range(3): abs_drift_trialspeed_avg[i] = abs_drift_trialspeed_avg[abs_drift_metric].map(lambda j: j[i])
age_bin_points = [7.5, 15, 25, 35, 45, 59]
fig, ax = plt.subplots(1, 3, figsize = standard_fig, sharey = True)
for t, trial_speed in enumerate(["SlowedDown","NoChange","SpedUp"]):
    data_to_plot = mean_results.loc[mean_results.trial_speed == trial_speed]
    age_bin_data_to_plot = abs_drift_trialspeed_avg.set_index("trial_speed").loc[trial_speed].sort_values("age_bin")
    ax[t].scatter(data_to_plot["age"], data_to_plot[abs_drift_metric], s = 50, color = "lightgray",
                    edgecolor="black", alpha = .3)
    ax[t].errorbar(age_bin_points, age_bin_data_to_plot[1].values, yerr = np.array([age_bin_data_to_plot[1]-age_bin_data_to_plot[0], age_bin_data_to_plot[2]-age_bin_data_to_plot[1]]),
                color = "black", linewidth = 3)
    ax[t].set_ylim(0,30)
    ax[t].tick_params(labelsize = 14)
    ax[t].set_xlabel("Age", fontsize = 16)
    ax[t].yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
    ax[t].set_title({"NoChange":"No Change","SlowedDown":"Slowed Down","SpedUp":"Sped Up"}[trial_speed], fontsize = 16)
ax[t].tick_params(labelsize = 14)
ax[0].set_ylabel("Absolute Drift\n(ITI Percent Change)", fontsize = 16, multialignment = "center", labelpad = 15)
fig.tight_layout()
fig.savefig(stats_plots + "abs_drift_age_trialspeed.png")
plt.close()

## Age Effect (Combined)
abs_drift_avg =  mean_results.groupby(["age_bin"]).agg({abs_drift_metric:
                                                    lambda values: tuple(bootstrap_ci(values, sample_percent = 75))}).reset_index()
for i in range(3): abs_drift_avg[i] = abs_drift_avg[abs_drift_metric].map(lambda j: j[i])
fig, ax = plt.subplots(1, 1, figsize = standard_fig, sharey = True)
ax.scatter(mean_results["age"], mean_results[abs_drift_metric], s = 50, color = "lightgray",
                edgecolor="black", alpha = .3)
ax.errorbar(age_bin_points, abs_drift_avg[1].values, yerr = np.array([abs_drift_avg[1]-abs_drift_avg[0], abs_drift_avg[2]-abs_drift_avg[1]]),
              color = "black", linewidth = 3)
ax.set_ylim(0,30)
ax.tick_params(labelsize = 14)
ax.set_xlabel("Age", fontsize = 16)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: "{}%".format(x)))
ax.tick_params(labelsize = 14)
ax.set_ylabel("Absolute Drift\n(ITI Percent Change)", fontsize = 16, multialignment = "center", labelpad = 15)
fig.tight_layout()
fig.savefig(stats_plots + "abs_drift_age.png")
plt.close()

################################################################################
### Variability Analysis
################################################################################

## Choose Variability Metric
sync_cv_metric = "met_cv_last5"
cont_cv_metric = "nomet_cv_last5"

## Fit Mixed LM Model for Sync
sync_cv_form = "{} ~ C(gender) + age + preferred_period + trial_speed + C(musical_experience) + C(healthy)".format(sync_cv_metric)
sync_cv_model = smf.mixedlm(sync_cv_form, data = mean_results, groups = mean_results["subject"]).fit(reml=True)
sync_cv_summary, sync_cv_wald = summarize_lm_model(sync_cv_model)

## Fit Mixed LM Model for Continuation
cont_cv_form = "{} ~ C(gender) + age + preferred_period + trial_speed + C(musical_experience) + C(healthy)".format(cont_cv_metric)
cont_cv_model = smf.mixedlm(cont_cv_form, data = mean_results, groups = mean_results["subject"]).fit(reml=True)
cont_cv_summary, cont_cv_wald = summarize_lm_model(cont_cv_model)


"""
- Trial speed, musical experience, age, and preferred period all have significant effects for synchronization
- Musical experience, age have significant effects for continuation
"""

## Trial Speed
sync_cv_trial_speed_avg = mean_results.groupby(["trial_speed"]).agg({sync_cv_metric: lambda values: tuple(bootstrap_ci(values, sample_percent = 75))}).reset_index()
cont_cv_trial_speed_avg = mean_results.groupby(["trial_speed"]).agg({cont_cv_metric: lambda values: tuple(bootstrap_ci(values, sample_percent = 75))}).reset_index()
for i in range(3): sync_cv_trial_speed_avg[i] = sync_cv_trial_speed_avg[sync_cv_metric].map(lambda j: j[i])
for i in range(3): cont_cv_trial_speed_avg[i] = cont_cv_trial_speed_avg[cont_cv_metric].map(lambda j: j[i])
fig, ax = plt.subplots(1, 1, figsize = standard_fig, sharey = True, sharex = True)
for t, trial_speed in enumerate(["SlowedDown","NoChange","SpedUp"]):
    sync_data_to_plot = sync_cv_trial_speed_avg.set_index("trial_speed").loc[trial_speed]
    cont_data_to_plot = cont_cv_trial_speed_avg.set_index("trial_speed").loc[trial_speed]
    ax.bar(0.025 + t, sync_data_to_plot[1], yerr = np.array([sync_data_to_plot[1]-sync_data_to_plot[0], sync_data_to_plot[2]-sync_data_to_plot[1]]).reshape(-1,1),
                color = "white", edgecolor = "black", width = .475, label = "Synchronization" if t == 0 else "", align = "edge")
    ax.bar(0.5 + t, cont_data_to_plot[1], yerr = np.array([cont_data_to_plot[1]-cont_data_to_plot[0], cont_data_to_plot[2]-cont_data_to_plot[1]]).reshape(-1,1),
                color = "lightgray", edgecolor = "black", width = .475, label = "Continuation" if t == 0 else "", align = "edge")
ax.legend(loc = "upper left", fontsize = 14, frameon = True, facecolor = "white")
ax.set_ylim(0,0.065)
ax.set_xticks(np.arange(3)+.5);
ax.set_xticklabels(["Slowed Down","No Change","Sped Up"])
ax.tick_params(labelsize = 14)
ax.set_xlabel("Trial Condition", fontsize = 16, labelpad = 15)
ax.set_ylabel("Coefficient of Variation", fontsize = 16, labelpad = 15)
fig.tight_layout()
fig.savefig(stats_plots + "variability_trialspeed.png")
plt.close()

## Musical Experience
cv_music_avg = mean_results.groupby(["trial_speed","musical_experience"]).agg({sync_cv_metric: lambda values: tuple(bootstrap_ci(values, sample_percent = 75)),
                                                                               cont_cv_metric: lambda values: tuple(bootstrap_ci(values, sample_percent = 75))}).reset_index()
for met in [sync_cv_metric, cont_cv_metric]:
    for i in range(3): cv_music_avg["{}_{}".format(met,i)] = cv_music_avg[met].map(lambda j: j[i])
fig, ax = plt.subplots(2, 1, figsize = standard_fig, sharey = True, sharex = True)
for t, trial_speed in enumerate(["SlowedDown","NoChange","SpedUp"]):
    for me in range(2):
        data_to_plot = cv_music_avg.set_index(["trial_speed","musical_experience"]).loc[trial_speed, me]
        for m, met in enumerate([sync_cv_metric, cont_cv_metric]):
            ax[m].bar(0.025 + t + me*.475, data_to_plot["{}_1".format(met)], width = .475,
                        yerr = np.array([data_to_plot["{}_1".format(met)]-data_to_plot["{}_0".format(met)],
                                    data_to_plot["{}_2".format(met)]-data_to_plot["{}_0".format(met)]]).reshape(-1,1),
                        color = "white" if me == 0 else "lightgray", edgecolor = "black", align = "edge",
                        label = {1:"Has Musical Experience",0:"No Musical Experience"}[me] if t == 0 else "")
ax[1].set_xticks(np.arange(3)+.5); ax[1].set_xticklabels(["Slowed Down", "No Change","Sped Up"])
ax[1].set_xlabel("Trial Condition", fontsize = 16, labelpad = 15)
ax[0].set_title("Synchronziation", fontsize = 16); ax[1].set_title("Continuation", fontsize = 16)
for a in ax:
    a.set_ylabel("Coefficient of\nVariation", fontsize = 16, labelpad = 15)
    a.tick_params(labelsize = 14)
    a.legend(loc = "upper right", fontsize = 12, frameon = True, facecolor = "white")
    a.set_ylim(0,0.1)
fig.tight_layout()
fig.subplots_adjust(top = .9)
fig.savefig(stats_plots + "variability_musicalexperience_trialspeed.png")
plt.close()

## Age
age_bin_var_avg = mean_results.groupby(["trial_speed","age_bin"]).agg({sync_cv_metric: lambda values: tuple(bootstrap_ci(values, sample_percent = 75)),
                                                         cont_cv_metric: lambda values: tuple(bootstrap_ci(values, sample_percent = 75))}).reset_index()
for met in [sync_cv_metric, cont_cv_metric]:
    for i in range(3): age_bin_var_avg["{}_{}".format(met,i)] = age_bin_var_avg[met].map(lambda j: j[i])
fig, ax = plt.subplots(1, 2, figsize = standard_fig, sharey = True)
for m, met in enumerate([sync_cv_metric, cont_cv_metric]):
    for t, (trial_speed, label) in enumerate(zip(["SlowedDown","NoChange","SpedUp"],["Slowed Down", "No Change", "Sped Up"])):
        scatter_data_to_plot = mean_results.loc[mean_results.trial_speed == trial_speed]
        avg_data_to_plot = age_bin_var_avg.loc[age_bin_var_avg.trial_speed == trial_speed]
        ax[m].scatter(scatter_data_to_plot["age"],scatter_data_to_plot[met], s = 50, color = "lightgray", edgecolor = "black",
                    alpha = 0.5, marker = {0:"o",1:"x", 2:"^"}[t])
        ax[m].errorbar(age_bin_points, avg_data_to_plot["{}_1".format(met)],
                        yerr = np.array([(avg_data_to_plot["{}_1".format(met)] - avg_data_to_plot["{}_0".format(met)]).values,
                        (avg_data_to_plot["{}_2".format(met)] - avg_data_to_plot["{}_1".format(met)]).values]),
                        color = "black", linewidth = 2, linestyle = {0:":",1:"--",2:"-"}[t])
    ax[m].set_xlabel("Age", fontsize = 16, labelpad = 15)
    ax[m].tick_params(labelsize = 14)
ax[0].set_ylabel("Coefficient of Variation", fontsize = 16, labelpad = 15)
ax[0].set_title("Synchronization", fontsize = 16)
ax[1].set_title("Continuation", fontsize = 16)
fig.tight_layout()
fig.savefig(stats_plots + "variability_age_trialspeed.png")
plt.close()

## Preferred Period
fig, ax = plt.subplots(1, 3, figsize = standard_fig, sharey = True)
for t, trial_speed in enumerate(["SlowedDown","NoChange","SpedUp"]):
    data_to_plot = mean_results.loc[mean_results.trial_speed == trial_speed]
    ax[t].scatter(data_to_plot["preferred_period"],data_to_plot[sync_cv_metric], color = "lightgray", edgecolor = "black",
                    alpha = .5, s = 50)
    exp_fit = ExponentialFit()
    exp_fit.fit(data_to_plot["preferred_period"], data_to_plot[sync_cv_metric])
    p_sorted = np.array(sorted(data_to_plot["preferred_period"].unique()))
    ax[t].plot(p_sorted, exp_fit.predict(p_sorted), color = "black", linewidth = 2)
    ax[t].tick_params(labelsize = 14)
    ax[t].set_xlabel("Preferred Period (ms)", fontsize = 16, labelpad = 15)
ax[0].set_ylabel("Coefficient of Variation", fontsize = 16, labelpad = 15)
for t, title in enumerate(["Slowed Down", "No Change", "Sped Up"]):
    ax[t].set_title(title, fontsize = 16)
fig.tight_layout()
fig.savefig(stats_plots + "variability_sync_preferredperiod.png")
plt.close()
