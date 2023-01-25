library(tidyverse)
library(tidymodels)
library(ranger)

library(here)

sales_split <- read_csv(here("data","sliced_s01e03","train.csv")) %>% 
  initial_split(prop = 0.9)
  
sales_test <- testing(sales_split)  
sales_train <- training(sales_split)  

sales_cv <- rsample::vfold_cv(sales_train, v = 10)  

# Pre-processing/recipes --------------------------------------------------


first_rec <- recipe(profit ~ id + sales + discount + city + quantity + category + sub_category,
                    data = sales_train) %>% 
  update_role(id, new_role = "id") %>% 
  step_log(sales) %>% 
  step_other(city) %>% 
  step_dummy(all_nominal_predictors())

#first_rec %>% prep %>% juice

# Random forest model -----------------------------------------------------
## Model spec --------------------------------------------------------------

linear_reg_glmnet_spec <-
  linear_reg(penalty = tune(), mixture = tune()) %>%
  set_engine('glmnet')

rand_forest_ranger_spec <-
  rand_forest(trees = 1000, mtry = tune(), min_n = tune()) %>%
  set_engine('ranger') %>%
  set_mode('regression')



## Tune params -----------------------------------------------------------


mset <- metric_set(rmse)

cgrid <- control_grid(verbose = TRUE,save_pred = TRUE)

rf_grid <- grid_latin_hypercube(mtry(c(4,15)), min_n(c(2, 10)),size = 10)

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

best_mn_log_l <- select_best(rand_wf, "mn_log_loss")

final_rf <-  workflow() %>% 
  add_model(finalize_model(rand_forest_ranger_spec,
                 select_best(rf_tune, "rmse"))) %>% 
  add_recipe(first_rec) %>% 
  fit(read_csv(here("data","sliced_s01e03","train.csv")))

final_rf %>%
  collect_metrics()

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


# Linear model
usemodels::use_glmnet(profit ~ sales + discount + city + quantity + category + sub_category,
                      data = train)


glmnet_recipe <- 
  recipe(formula = sales ~ discount + city + quantity + category + sub_category, 
         data = train) %>% 
  step_zv(all_predictors()) %>% 
  step_normalize(all_numeric_predictors()) 

glmnet_spec <- 
  linear_reg(penalty = tune(), mixture = tune()) %>% 
  set_mode("regression") %>% 
  set_engine("glmnet") 

glmnet_workflow <- 
  workflow() %>% 
  add_recipe(glmnet_recipe) %>% 
  add_model(glmnet_spec) 

glmnet_grid <- tidyr::crossing(penalty = 10^seq(-6, -1, length.out = 20), mixture = c(0.05, 
                                                                                      0.2, 0.4, 0.6, 0.8, 1)) 

glmnet_tune <- 
  tune_grid(glmnet_workflow, resamples = stop("add your rsample object"), grid = glmnet_grid) 



