library(tidyverse)
library(tidymodels)
library(glmnet)
library(ranger)

# David Robinson, Sliced Episode 4 - https://github.com/dgrtwo/data-screencasts/blob/master/ml-practice/ep4.Rmd

# Load data ---------------------------------------------------------------

test <- read_csv("data/test.csv")
train <- read_csv("data/train.csv") %>% 
  mutate(rain_tomorrow = if_else(rain_tomorrow == 1, "IsRaining","NotRaining") %>% factor())

train_folds <- vfold_cv(train,v = 5,strata = rain_tomorrow)

# Specify model -------------------------------------------------------------

rf_spec <- rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()) %>% 
  set_engine(engine = "ranger") %>% 
  set_mode("classification")

# Recipe ------------------------------------------------------------------

prep_bake <- function(wf){
  wf %>% prep() %>% bake(new_data = NULL)
}

weather_recipe2 <- recipe(rain_tomorrow ~ rain_today + humidity3pm + humidity9am + date, data = train) %>% 
  step_impute_median(all_numeric_predictors(),-humidity3pm) %>% 
  step_impute_linear(humidity3pm,impute_with = imp_vars(humidity9am)) %>% 
  step_date(date, features = c("month", "year"),keep_original_cols = F) %>% 
  step_mutate(date_year = factor(date_year))
  
# Tuning ------------------------------------------------------------------

rf_grid <- grid_regular(
  mtry(range = c(3,5)),
  trees(range = c(100,1250)),
  min_n(range = c(4, 8)),
  levels = 2
)


# Workflow ----------------------------------------------------------------

rand_wf <- workflow() %>%
  add_recipe(weather_recipe2) %>%
  add_model(rf_spec) %>% 
  tune_grid(train_folds,
            grid = rf_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(roc_auc,specificity,sensitivity,mn_log_loss))

autoplot(rand_wf)