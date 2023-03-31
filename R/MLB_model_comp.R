library(tidyverse)
library(here)
library(tidymodels)
library(ranger)
library(xgboost)

dataset <- read_csv(here("data", "MLB-pitch-data", "train.csv")) %>% 
  mutate(is_strike = if_else(is_strike == 1, "stike", "no_strike"))
holdout <- read_csv(here("data", "MLB-pitch-data", "test.csv")) 

mlb_split <- initial_split(dataset)
test <- testing(mlb_split)
train <- training(mlb_split)


# Build recipe ------------------------------------------------------------

mlb_recipe_simple <- 
  recipe(formula = is_strike ~ uid + 
           # Categorical
           pitch_type + if_fielding_alignment + of_fielding_alignment + 
           # Logical
           on_3b + on_1b + on_2b +
           # Numeric
           balls + strikes + outs_when_up + plate_x + plate_z + release_spin_rate + 
           release_speed + sz_top + sz_bot,
         data = train) %>% 
  update_role(uid, new_role = "id") %>% 
  step_zv(all_predictors()) %>% 
  step_mutate(inZone_low = plate_z - sz_bot,
               inZone_high = plate_z - sz_top) %>% 
  step_rm(sz_bot, sz_top) %>% 
  step_bin2factor(on_1b, on_2b, on_3b) %>% 
  step_novel(pitch_type, if_fielding_alignment, of_fielding_alignment) %>% 
  step_unknown(pitch_type, if_fielding_alignment, of_fielding_alignment) %>% 
  step_other(pitch_type, if_fielding_alignment, of_fielding_alignment, threshold = 0.01) %>% 
  step_dummy(all_nominal_predictors())

# p = 27

## Engineered features

mlb_recipe_engineered <- 
  recipe(formula = is_strike ~ uid + 
           # Categorical
           pitch_type + if_fielding_alignment + of_fielding_alignment + stand + p_throws +
           # Logical
           on_3b + on_1b + on_2b +
           # Numeric
           balls + strikes + outs_when_up + plate_x + plate_z + release_spin_rate + 
           release_speed + sz_top + sz_bot + release_pos_x + release_pos_y + release_pos_z, 
         data = train) %>% 
  update_role(uid, new_role = "id") %>% 
  step_zv(all_predictors()) %>% 
  step_mutate(inZone_low = plate_z - sz_bot,
              inZone_high = plate_z - sz_top,
              plate_x = if_else(stand == "L", -plate_x, plate_x),
              release_pos_x = if_else(p_throws == "L", -release_pos_x, release_pos_x)) %>% 
  step_rm(sz_bot, sz_top) %>% 
  step_bin2factor(on_1b, on_2b, on_3b) %>% 
  step_other(pitch_type, if_fielding_alignment, of_fielding_alignment, threshold = 0.01) %>% 
  step_novel(pitch_type, if_fielding_alignment, of_fielding_alignment) %>% 
  step_unknown(pitch_type, if_fielding_alignment, of_fielding_alignment) %>% 
  step_dummy(all_nominal_predictors())

# p = 32

# Model Training ----------------------------------------------------------

mset <- metric_set(mn_log_loss)
control <- control_grid(verbose = TRUE, save_pred = TRUE)

mlb_cv_small <- train %>% 
  sample_n(20000) %>% 
  vfold_cv(strata = is_strike, v = 5)

xgboost_spec1 <- 
  boost_tree(
    mode = "classification",
    trees = tune(),
    mtry = tune(),
    learn_rate = 0.02)
#   min_n = 10,
#   tree_depth = 5,
#   learn_rate = 0.1,
#   loss_reduction = 0
# )

grid_step1 <- crossing(trees = seq(200, 600, 100), mtry = seq(6, 15, 3))


xgboost_tune1 <-
  tune_grid(
    object = xgboost_spec1,
    preprocessor = mlb_recipe_simple,
    resamples = mlb_cv_small,
    grid = grid_step1,
    metrics = mset,
    control = control
  )

beepr::beep(8)
autoplot(xgboost_tune1)

xgboost_tune1 %>%
  collect_metrics() %>% 
  ggplot(aes(x = trees, y = mean, col = factor(mtry), group = .config)) +
  geom_point() +
  geom_line()

#mtry = 12, trees = 500


# Engineered --------------------------------------------------------------

grid_step2 <- crossing(trees = seq(200, 700, 100), mtry = seq(6, 20, 3))

xgboost_tune2 <-
  tune_grid(
    object = xgboost_spec1,
    preprocessor = mlb_recipe_engineered,
    resamples = mlb_cv_small,
    grid = grid_step2,
    metrics = mset,
    control = control
  )

beepr::beep(8)

autoplot(xgboost_tune2)

select_best(xgboost_tune2)

# Workflow set ------------------------------------------------------------

mlb_cv_medium <- train %>% 
  sample_frac(size = 0.5) %>% 
  vfold_cv(strata = is_strike, v = 10)

xgboost_spec_tuned <- 
  boost_tree(
    mode = "classification",
    trees = 600,
    mtry = 12,
    learn_rate = 0.02)

randomForest_spec <-
  rand_forest(
    mode = "classification",
    engine = 'ranger',
    mtry = 6,
    min_n = 5,
    trees = 1000
  )

recipe_trials <-
  workflow_set(
    preproc = list(simple = mlb_recipe_simple, engineered = mlb_recipe_engineered),
    models = list(xgb = xgboost_spec_tuned, rf = randomForest_spec),
    cross = TRUE
  ) %>%
  workflow_map(
    "fit_resamples",
    resamples = mlb_cv_medium,
    metrics = mset,
    control = control
  )

recipe_trials %>% autoplot()

# # A tibble: 4 × 9
# wflow_id       .config              preproc model       .metric     .estimator  mean     n  std_err
# <chr>          <chr>                <chr>   <chr>       <chr>       <chr>      <dbl> <int>    <dbl>
#   1 simple_xgb     Preprocessor1_Model1 recipe  boost_tree  mn_log_loss binary     0.153    10 0.00110 
# 2 simple_rf      Preprocessor1_Model1 recipe  rand_forest mn_log_loss binary     0.165     9 0.00124 
# 3 engineered_xgb Preprocessor1_Model1 recipe  boost_tree  mn_log_loss binary     0.151    10 0.000988
# 4 engineered_rf  Preprocessor1_Model1 recipe  rand_forest mn_log_loss binary     0.174    10 0.00109 

