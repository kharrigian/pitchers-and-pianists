library(lme4)

#########################
### Load Data
#########################

# Set Directory
setwd("/Users/Keith/Documents/ActionLab/MuseumOfScience/")

# Processed Results File
datafile <- "./data/transformed_subset.csv"
data <- read.csv(datafile)

# Encode Variable Types
data$trial <- as.numeric(data$trial)
data$trial_speed <- as.factor(data$trial_speed)
data$gender <- as.factor(data$gender)
data$age <- as.numeric(data$age)
data$speed_occurrence <- as.numeric(data$speed_occurrence)
data$musical_experience <- as.factor(data$musical_experience)
data$sport_experience <- as.factor(data$sport_experience)

# Dependent Variables
data$abs_met_sync_error_last10 <- as.numeric(data$abs_met_sync_error_last10)
data$sync_error_transformed <- as.numeric(data$sync_error_transformed)
data$abs_nomet_sync_error_last20 <- as.numeric(data$abs_nomet_sync_error_last20)
data$cont_error_transformed <- as.numeric(data$cont_error_transformed)
data$met_cv_last10 <- as.numeric(data$met_cv_last10)
data$nomet_cv_last20 <- as.numeric(data$nomet_cv_last20)
data$var_met_transformed <- as.numeric(data$var_met_transformed)
data$var_nomet_transformed <- as.numeric(data$var_nomet_transformed)
data$nomet_drift <- as.numeric(data$nomet_drift)


#########################
### Synchronization Error
#########################


model_formula <- "sync_error_transformed ~ 
                      preferred_period + 
                      gender + age + trial_speed + trial +   
                      musical_experience + sport_experience + 
                      (1 + speed_occurence | subject) + (1 + trial | subject) +
                      (1 | trial_speed)
                      "

mod1 <- lmer(model_formula, data = data)
summary(mod1)
