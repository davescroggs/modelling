## Fit model to SBA data

library(tidyverse)
library(tidymodels)
library(scales)
library(here)
library(vip)

SBA_train <- read_csv(here("data/sliced-s01e12/train.csv"))
SBA_test <- read_csv(here("data/sliced-s01e12/test.csv"))

SBA_folds <- rsample::vfold_cv(SBA_train,v = 5)



# Model spec --------------------------------------------------------------

sba_XGB <- boost_tree() %>%
  set_engine("xgboost") %>%
  set_mode("regression")

# Pre-processing ----------------------------------------------------------

XGB_rec <- recipe(default_amount ~ .,
                  data = SBA_train) %>% 
  #step_log(default_amount,offset = 1) %>% 
  step_other(Bank,threshold = 0.003) %>% 
  step_other(NewExist,FranchiseCode,UrbanRural) %>% 
  step_impute_mode(NewExist) %>% 
  update_role(LoanNr_ChkDgt,Name,new_role = "id") %>%
  step_rm(NAICS,Zip,City)
  #step_mutate(UrbanRural = factor(UrbanRural,levels = c("0","1","2"))) %>% 
  step_dummy(all_nominal_predictors(),one_hot = T)


# Workflow ----------------------------------------------------------------

XGB_wf <- workflow() %>% 
  add_model(sba_XGB) %>% 
  add_recipe(XGB_rec)

XGB_wfs <-
  workflow_set(list(XGB_rec, XGB_no_dummy),
               models = list(sba_XGB),
               cross = T)
  

# Fit resamples -----------------------------------------------------------

XGB_rs <- fit_resamples(XGB_wf,
                        SBA_folds,
                        control = control_resamples(save_pred = TRUE),
                        metrics = metric_set(mae))



# Tune XGBoost ------------------------------------------------------------
# Racecar?




# Workflow sets ---------------------------------------------------------------



XGB_rs <- XGB_wfs %>% 
  workflow_map(fn = "fit_resamples",
               resamples = SBA_folds,
               control = control_resamples(save_pred = TRUE),
               metrics = metric_set(mae))


# Evaluate ----------------------------------------------------------------


collect_metrics(XGB_rs)


# Last fit ----------------------------------------------------------------

last_fit(XGB_wfm)

?last_fit
