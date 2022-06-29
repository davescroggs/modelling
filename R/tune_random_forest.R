library(tidyverse)
library(tidymodels)
library(glmnet)
library(ranger)

doParallel::registerDoParallel(cores = 4)

# David Robinson, Sliced Episode 4 - https://github.com/dgrtwo/data-screencasts/blob/master/ml-practice/ep4.Rmd

# Load data ---------------------------------------------------------------

validation <- read_csv("data/test.csv")

weather_split <- read_csv("data/train.csv") %>% 
  mutate(rain_tomorrow = if_else(rain_tomorrow == 1, "IsRaining","NotRaining") %>% factor()) %>% 
  initial_split(prop = 0.75,strata = rain_tomorrow)

train <- training(weather_split)
test <- testing(weather_split)

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

## Check pre-processed dataset
  
weather_recipe2 %>% 
  prep_bake

# Tuning ------------------------------------------------------------------

rf_grid <- grid_regular(
  mtry(range = c(2,5)),
  trees(range = c(1000,1500)),
  min_n(range = c(4, 8)),
  levels = 3
)


# Workflow ----------------------------------------------------------------

small_folds <- 
  train %>% 
  sample_frac(0.15) %>% 
  vfold_cv(v = 5,strata = rain_tomorrow)

rand_wf <- workflow() %>%
  add_recipe(weather_recipe2) %>%
  add_model(rf_spec) %>% 
  tune_grid(small_folds,
            grid = rf_grid,
            control = control_grid(save_pred = TRUE),
            metrics = metric_set(mn_log_loss))

collect_metrics(rand_wf) %>% 
  arrange(mean)


# Final model --------------------------------------------------------------

best_mn_log_l <- select_best(rand_wf, "mn_log_loss")

final_rf <- finalize_model(
  rf_spec,
  best_mn_log_l
)

final_wf <- workflow() %>%
  add_recipe(weather_recipe2) %>%
  add_model(final_rf)

final_res <- final_wf %>%
  last_fit(weather_split,metrics = metric_set(mn_log_loss))

final_res %>%
  collect_metrics()
