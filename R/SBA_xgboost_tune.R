# Tune XGBoost
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










xgb_spec <- boost_tree(
  trees = 1000, 
  tree_depth = tune(), min_n = tune(), 
  loss_reduction = tune(),                     ## first three: model complexity
  sample_size = tune(), mtry = tune(),         ## randomness
  learn_rate = tune(),                         ## step size
) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

xgb_spec

xgb_grid <- grid_latin_hypercube(
  tree_depth(),
  min_n(),
  loss_reduction(),
  sample_size = sample_prop(),
  finalize(mtry(), vb_train),
  learn_rate(),
  size = 30
)