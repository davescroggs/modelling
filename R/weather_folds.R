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


# EDA ---------------------------------------------------------------------

train %>% 
  group_by(rain_tomorrow) %>% 
  summarise(n = n()) %>% 
  mutate(pct = n/sum(n))

train %>%
    mutate(rainfall = log2(rainfall + 1)) %>%
    gather(metric, value, min_temp, max_temp, rainfall, contains("speed"), contains("humidity"), contains("pressure"), contains("cloud"), contains("temp")) %>% 
    group_by(metric) %>%
    roc_auc(rain_tomorrow, value, event_level = "second") %>%
    arrange(desc(.estimate)) %>%
    mutate(metric = fct_reorder(metric, .estimate)) %>%
    ggplot(aes(.estimate, metric)) +
    geom_point() +
    geom_vline(xintercept = .5) +
    labs(x = "AUC in positive direction",
         title = "How predictive is each linear predictor by itself?",
         subtitle = ".5 is not predictive at all; <.5 means negatively associated with rain, >.5 means positively associated")

train %>%
    mutate(year = year(date)) %>% 
    group_by(year,rain_tomorrow) %>% 
    summarise(n = n()) %>% 
    mutate(pct = n/sum(n)) %>% 
    filter(rain_tomorrow == "IsRaining") %>% 
    ggplot(aes(year,pct)) +
    geom_line() +
    geom_smooth(method = lm, formula = y ~ splines::bs(x, 5), se = FALSE) +
    ylim(0,0.35)

# Specify model -------------------------------------------------------------

rf_spec <- rand_forest(
  mtry = tune(),
  trees = tune(),
  min_n = tune()) %>% 
    set_engine(engine = "ranger") %>% 
    set_mode("classification")

glm_spec <- logistic_reg(penalty = tune(), mixture = 1) %>% 
    set_engine(engine = "glmnet") %>% 
    set_mode("classification")

# Recipe ------------------------------------------------------------------

weather_recipe <- recipe(rain_tomorrow ~ rain_today + humidity3pm, data = train) %>% 
    step_impute_mean(rain_today,humidity3pm)

weather_recipe2 <- recipe(rain_tomorrow ~ rain_today + humidity3pm + humidity9am + date, data = train) %>% 
    step_impute_linear(humidity3pm,impute_with = imp_vars(humidity9am)) %>% 
    step_impute_median(all_numeric_predictors()) %>% 
    step_date(date, features = c("month", "year"),keep_original_cols = F) %>% 
    step_num2factor(date_year, levels = as.character(2007:2017))


# Tuning ------------------------------------------------------------------

set.seed(345)
tune_res <- tune_grid(
  rf_spec,
  resamples = trees_folds,
  grid = 20
)


# Workflow ----------------------------------------------------------------

rand_wf <- workflow() %>%
    add_recipe(weather_recipe2) %>%
    add_model(rf_spec)

log_ref_wf <- workflow() %>%
    add_recipe(weather_recipe2) %>%
    add_model(glm_spec) %>% 
    tune_grid(val_set,
              grid = lr_reg_grid,
              control = control_grid(save_pred = TRUE),
              metrics = metric_set(roc_auc))

# Fit model ---------------------------------------------------------------

ctrl <- control_resamples(extract = function(x) extract_fit_parsnip(x) %>% tidy(),
                          save_pred = TRUE)

fit_rf <- rand_wf %>% 
  fit_resamples(resamples = train_folds, control = ctrl)

# Logistic regression

lr_reg_grid <- tibble(penalty = 10^seq(-4, -1, length.out = 30))

fit_lr <- log_ref_wf %>%  fit_resamples(resamples = train_folds, control = ctrl)
# Evaluate performance ----------------------------------------------------

fit_rf %>% 
    mutate(f_meas = map_dbl(.predictions,~f_meas_vec(truth = .x$rain_tomorrow,estimate = .x$.pred_class)),
           accuracy = map_dbl(.predictions,~accuracy_vec(truth = .x$rain_tomorrow,estimate = .x$.pred_class)),
           mn_log_loss = map_dbl(.predictions,~mn_log_loss_vec(truth = .x$rain_tomorrow,estimate = .x$.pred_IsRaining))) %>% 
    summarise(mean = mean(mn_log_loss))


fit_r2 %>% 
    mutate(f_meas = map_dbl(.predictions,~f_meas_vec(truth = .x$rain_tomorrow,estimate = .x$.pred_class)),
           accuracy = map_dbl(.predictions,~accuracy_vec(truth = .x$rain_tomorrow,estimate = .x$.pred_class)),
           mn_log_loss = map_dbl(.predictions,~mn_log_loss_vec(truth = .x$rain_tomorrow,estimate = .x$.pred_IsRaining))) %>% 
    summarise(mean = mean(mn_log_loss))


?finalize_workflow


workflow_set()