library(tidyverse)
library(tidymodels)
library(glmnet)
library(ranger)



# Load data ---------------------------------------------------------------

test <- read_csv("data/test.csv")
train <- read_csv("data/train.csv") %>% 
    mutate(rain_tomorrow = if_else(rain_tomorrow == 1, "IsRaining","NotRaining") %>% factor()) %>% 
    filter(!(is.na(rain_today) | is.na(humidity3pm)))

# Specify model -------------------------------------------------------------
    
rf_spec <- rand_forest() %>% 
    set_engine(engine = "ranger") %>% 
    set_mode("classification")


# Recipe1 ------------------------------------------------------------------

weather_recipe1 <- recipe(rain_tomorrow ~ rain_today + humidity3pm + date, data = train) %>% 
    step_naomit(rain_today, humidity3pm, date) %>% 
    step_date(date, features = c("month", "year"),keep_original_cols = F)
    
weather_recipe2 <- recipe(rain_tomorrow ~ rain_today + humidity3pm + humidity9am + date, data = train) %>% 
    step_impute_median(humidity9am) %>% 
        step_impute_linear(humidity3pm,impute_with = imp_vars(humidity9am)) %>% 
        step_impute_median(humidity3pm,rain_today,humidity9am) %>% 
    step_date(date, features = c("month", "year"),keep_original_cols = F)

# Workflow ----------------------------------------------------------------

rand_wf <- workflow() %>%
    add_recipe(weather_recipe2) %>%
    add_model(rf_spec)

# Fit model ---------------------------------------------------------------


fit_rf <- fit(rand_wf,train)


# Evaluate performance ----------------------------------------------------

perf <- train %>% 
    {select(.,rain_tomorrow,rain_today,humidity3pm,humidity9am,date) %>% 
            bind_cols(predict(fit_rf, .,type = "prob")) %>% 
            bind_cols(predict(fit_rf, .))}
    
weather_perf_metrics <- metric_set(accuracy,f_meas)

conf_mat(perf,truth = rain_tomorrow,estimate = .pred_class)

weather_perf_metrics(perf,truth = rain_tomorrow,estimate = .pred_class) %>% 
    bind_rows(
mn_log_loss(perf,truth = rain_tomorrow,estimate = .pred_IsRaining))

roc_curve(perf, rain_tomorrow, .pred_IsRaining) %>% 
autoplot()

# Multiple formulas -------------------------------------------------------

library(workflowsets)
location_models <- workflow_set(preproc = location, models = list(lm = lm_model))
location_models

weather_recipe2 %>% prep() %>% bake(new_data = train) %>% 
    filter_all(any_vars(is.na(.)))


# Output results ----------------------------------------------------------

test %>% 
    transmute(id = id,
              rain_tomorrow = predict(fit_rf,test,type = "prob")$.pred_IsRaining) %>% 
    write_csv(file = "data/kaggle_submission.csv")
