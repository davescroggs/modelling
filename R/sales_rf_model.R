library(tidyverse)
library(tidymodels)
library(ranger)

library(here)

sales_split <- read_csv(here("data","sliced_s01e03","train.csv")) %>% 
  initial_split(prop = 0.9)
  
sales_test <- testing(sales_split)  
sales_train <- training(sales_split)  

sales_cv <- rsample::vfold_cv(sales_train, v = 5)  

# Random forest model -----------------------------------------------------
# Pre-processing/recipes --------------------------------------------------


first_rec <- recipe(profit ~ id + sales + discount + city + quantity + category + sub_category,
                    data = sales_train) %>% 
  update_role(id, new_role = "id") %>% 
  step_log(sales) %>% 
  step_other(city, threshold = 40) %>% 
  step_dummy(all_nominal_predictors())

first_rec %>% prep %>% juice

## Model spec --------------------------------------------------------------

rand_forest_ranger_spec <-
  rand_forest(trees = 1000, mtry = tune(), min_n = tune()) %>%
  set_engine('ranger') %>%
  set_mode('regression')



## Tune params -----------------------------------------------------------


mset <- metric_set(rmse)

cgrid <- control_grid(verbose = TRUE,save_pred = TRUE)

rf_grid <- grid_latin_hypercube(mtry(c(10,20)), min_n(c(2, 10)),size = 10)

rf_tune <- tune_grid(rand_forest_ranger_spec, first_rec, sales_cv,
                     grid = rf_grid,
                     control = cgrid,
                     metrics = mset)

rf_tune %>% autoplot()

rf_tune %>% 
  collect_metrics()


## Finalise model ----------------------------------------------------------

rf_tune %>% 
  select_best()

final_rf <-  workflow() %>% 
  add_model(finalize_model(rand_forest_ranger_spec,
                 select_best(rf_tune, "rmse"))) %>% 
  add_recipe(first_rec) %>% 
  fit(read_csv(here("data","sliced_s01e03","train.csv")))

final_rf %>% 
  augment(read_csv(here("data","sliced_s01e03","test.csv")))

## Generate submission -----------------------------------------------------

library(broom)
predict_holdout <- function(wf) {
  wf %>%
    augment(read_csv(here("data","sliced_s01e03","test.csv"))) %>%
    select(id, profit = .pred)
}
augment.fit <- function(x, data, ...) {
  bind_cols(data, predict(x, data, ...))
}

final_rf %>% 
  predict_holdout() %>% 
  write_csv("data/sliced_s01e03/sample_sub.csv")


## Feature Importance ------------------------------------------------------
library(vip)

final_rf %>%
  extract_fit_engine() %>% 
  vip(geom = "point")

# XGBoost ------------------------------------------------------------

usemodels::use_xgboost(profit ~ id + sales + discount + city + quantity + category + sub_category,
                        data = sales_train, verbose = TRUE, tune = TRUE)

xgboost_recipe <- 
  recipe(formula = profit ~ id + sales + discount + city + quantity + category + 
           sub_category, data = sales_train) %>% 
  step_zv(all_predictors()) %>% 
  update_role(id, new_role = "id") %>% 
  step_log(sales) %>% 
  step_other(city, threshold = 40) %>% 
  step_dummy(all_nominal_predictors())

xgboost_spec <- 
  boost_tree(trees = tune(), min_n = tune(), tree_depth = tune(), learn_rate = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("xgboost")

xgb_grid <- grid_latin_hypercube(trees(c(400,1200)), min_n(c(2,40)), tree_depth(c(1,15)), learn_rate(c(.005, .01), trans = NULL), size = 10)

xgboost_workflow <- 
  workflow() %>% 
  add_recipe(xgboost_recipe) %>% 
  add_model(xgboost_spec) 

xgboost_tune <-
  tune_grid(xgboost_workflow, resamples = sales_cv,
            grid = xgb_grid, control = cgrid,
            metrics = mset)

xgboost_tune %>% 
  autoplot()

## Finalise model ----------------------------------------------------------

xgboost_tune %>% 
  select_best()

final_xgb <-  workflow() %>% 
  add_model(boost_tree(trees = 1000, min_n = 5, tree_depth = 5, learn_rate = 0.01) %>% 
              set_mode("regression") %>% 
              set_engine("xgboost")) %>% 
  add_recipe(xgboost_recipe) %>% 
  fit(read_csv(here("data","sliced_s01e03","train.csv")))

final_xgb %>% 
  predict_holdout() %>% 
  write_csv("data/sliced_s01e03/sample_sub.csv")

