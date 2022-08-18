library(tidyverse)
library(tidymodels)
library(ranger)
library(lubridate)

doParallel::registerDoParallel(cores = 2)

# David Robinson, Sliced Episode 4 - https://github.com/dgrtwo/data-screencasts/blob/master/ml-practice/ep4.Rmd
# Kaggle competition - 

# Load data ---------------------------------------------------------------

test <- read_csv("data/sliced-s01e04/test.csv")
train <- read_csv("data/sliced-s01e04/train.csv") %>% 
    mutate(rain_tomorrow = if_else(rain_tomorrow == 1, "IsRaining","NotRaining") %>% factor())

train_folds <- 
  vfold_cv(train,v = 5,strata = rain_tomorrow)

train_split <- initial_split(train,strata = rain_tomorrow,prop = 0.9)

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

rf_basic <- rand_forest() %>% 
  set_engine(engine = "ranger") %>% 
  set_mode("classification")

rf_spec <- rand_forest(
  mtry = tune(),
  trees = 1000,
  min_n = tune()) %>% 
    set_engine(engine = "ranger") %>% 
    set_mode("classification")

# Recipe ------------------------------------------------------------------

prep_juice <- function(x) prep(x) %>% juice()

weather_spline_rec <- 
  recipe(rain_tomorrow ~ date + rain_today,data = train) %>% 
  step_impute_median(all_numeric_predictors()) %>% 
  step_date(date,
            features = c("month", "week"),
            keep_original_cols = F,label = F) %>%
  #step_mutate(date_year = as.character(date_year) %>% factor(levels = as.character(2007:2017))) %>% 
  step_ns(date_week,deg_free = tune("deg week")) %>% 
  step_ns(date_month,deg_free = tune("deg month"))

weather_recipe <- recipe(rain_tomorrow ~ rain_today + humidity3pm + humidity9am +
                           date + pressure9am + pressure3pm + id,
                         data = train) %>%
  update_role(id,new_role = "id") %>% 
  step_impute_median(humidity9am) %>%
  step_impute_linear(humidity3pm, impute_with = imp_vars(humidity9am)) %>%
  step_impute_median(all_numeric_predictors()) %>%
  step_date(date,
            features = c("month", "year", "week"),
            keep_original_cols = F,label = FALSE) %>%
  step_mutate(date_year = as.character(date_year) %>% factor(levels = as.character(2007:2017))) %>% 
  step_ns(date_week,deg_free = 2) %>% 
  step_ns(date_month,deg_free = 8) %>% 
  step_dummy(date_year)

#Inspect the resulting processed training data
weather_recipe %>% 
  prep() %>% 
  juice()

# Tuning ------------------------------------------------------------------

small_train <- train %>% 
  sample_n(10000) %>% 
  vfold_cv(v = 5,strata = rain_tomorrow)

mset <- metric_set(mn_log_loss,f_meas)


# Tune splines ------------------------------------------------------------

df_vals <- 2:10
# A regular grid:
spline_grid <- expand.grid(`deg week` = df_vals, `deg month` = df_vals)

tune_splines <- tune_grid(
  rf_basic,
  preprocessor = weather_spline_rec,
  resamples = small_train,
  metrics = mset,
  grid = spline_grid,
  control = control_grid(verbose = TRUE)
)

autoplot(tune_splines)

collect_metrics(tune_splines)

select_best(tune_splines)

# Best splines
# `deg week` = 2, `deg month` = 8

# Tune RF params ----------------------------------------------------------

rf_wf <- workflow() %>% 
  add_recipe(weather_recipe) %>% 
  add_model(rf_spec)

rf_grid <- grid_max_entropy(
  mtry(range = c(2, 15)),
  min_n(range = c(6,15)),
  size = 10
)


set.seed(345)
tune_rf_r1 <- tune_grid(
  rf_spec,
  preprocessor = weather_recipe,
  resamples = small_train,
  metrics = mset,
  grid = rf_grid,
  control = control_grid(verbose = TRUE)
)

autoplot(tune_rf)

collect_metrics(tune_rf)



# Tune round 2 ------------------------------------------------------------

# Try 2-6 predictors (mtry), min_n 15-30 

rf_grid <- grid_regular(
  mtry(range = c(2, 6)),
  min_n(range = c(15,30)),
  levels = 5
)


set.seed(345)
tune_rf_r2 <- tune_grid(
  rf_spec,
  preprocessor = weather_recipe,
  resamples = small_train,
  metrics = mset,
  grid = rf_grid,
  control = control_grid(verbose = TRUE)
)

autoplot(tune_rf)

# mtry = 5, min_n = 30

# Fit model ---------------------------------------------------------------

ctrl <- control_resamples(extract = function(x) extract_fit_parsnip(x) %>%
                            tidy(),
                          save_pred = TRUE,
                          verbose = TRUE)

final_rf <- 
  finalize_workflow(rf_wf,select_best(tune_rf,metric = "mn_log_loss")) 

final_res <- final_wf %>%
  last_fit(train_split,metrics = mset)

# Evaluate performance ----------------------------------------------------

final_res %>%
  collect_metrics()

# Create sample submission ------------------------------------------------

final_res