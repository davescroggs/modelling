
# Model AFL kicking chains ------------------------------------------------

library(tidyverse)
library(tidymodels)


# Caculate pass angle
# Directly toward the goals is 0 degrees
pass_angle <- function(O1,O2,M1,M2){
  M = c(M1,M2)
  O = c(O1,O2)
  M = M - O
  # Position of the middle of the forward direction 
  # ie. middle of goals
  N = c(68,0) - O
  theta = atan2(N[2],N[1]) - atan2(M[2],M[1]) 
  return(theta * 180/pi)
}

# Calculate pass distance
calc_dist <- function(x0, y0, x1, y1){
  pass_dist = sqrt((x1-x0)^2 + (y1 - y0)^2)
  return(pass_dist)
}


# Load data ---------------------------------------------------------------

kick_raw <- read_csv("data/kick_rating_data.csv") %>% 
  filter(!desc_next %in% c("out_on_full_after_kick", "out_on_full"))


kicks <- 
  kick_raw %>%   
  select(home_team, away_team, playing_for, desc_last, x_norm:y_next1, threat:disposal) %>% 
  mutate(across(c(threat:disposal), as.factor),
         pass_angle = pmap_dbl(list(x_norm, y_norm, x_next1, y_next1),  pass_angle),
         pass_distance = pmap_dbl(list(x_norm, y_norm, x_next1, y_next1),  calc_dist),
         distance_from_goal_start = map2_dbl(x_norm, y_norm,  ~calc_dist(.x, .y, 68, 0)),
         distance_from_goal_end = map2_dbl(x_next1, y_next1,  ~calc_dist(.x, .y, 68, 0)),
         situation = 
         is_home = if_else(home_team == playing_for, "home", "away") %>% factor(levels = c("home", "away"))) %>% 
  select(-c(retention, disposal, home_team, away_team, playing_for))


# EDA ---------------------------------------------------------------------

kicks %>%
  select(threat, where(is.numeric)) %>% 
  pivot_longer(cols = -threat) %>% 
  group_by(name) %>%
  yardstick::roc_auc(truth = threat, value, event_level = "second") %>% 
  arrange(desc(.estimate)) %>%
  mutate(name = fct_reorder(name, .estimate)) %>%
  ggplot(aes(.estimate, name)) +
  geom_point() +
  geom_vline(xintercept = .5) +
  labs(x = "AUC",
       y = "Variable",
       title = "ROC AUC of numeric variables",
       subtitle = ".5 is not predictive at all; <.5 means negatively associated with stikes, >.5 means positively associated")

# Summarise categorical features
# Caclulate percentage of each level that ends in a goal
summarise_threat <- function(df) {
  df %>% 
    summarise(n_threat = sum(threat == 1),
              n = n(),
              .groups = "drop") %>% 
    arrange(desc(n)) %>% 
    mutate(pct_threat = n_threat/n,
           # Calculate uninformed prior
           low = qbeta(.025, n_threat + 0.5, n - n_threat + 0.5),
           high = qbeta(.975, n_threat + 0.5, n - n_threat + 0.5)) %>%
    mutate(pct = n / sum(n))
}

summary_plot <- function(df, grp){
  df %>% 
    mutate("{{grp}}" :=fct_reorder({{ grp }}, pct_threat)) %>% 
    ggplot(aes(x = pct_threat, y = {{grp}})) +
    geom_point(aes(size = n)) +
    geom_errorbar(aes(xmin = low, xmax = high), width = 0.5) +
    # Percentage of all kicks that end in goals
    geom_vline(xintercept = 0.228, linetype = "dashed") +
    scale_x_continuous(labels = percent)
}

# Lots of small features - could be worth lumping
kicks %>% 
  group_by(desc_last) %>% 
  summarise_threat() %>% 
  summary_plot(desc_last)

kicks %>% 
  mutate(desc_last = fct_lump_prop(desc_last, prop = 0.005)) %>%
  group_by(desc_last) %>% 
  summarise_threat() %>% 
  summary_plot(desc_last)

kicks %>% 
  group_by(is_home) %>% 
  summarise_threat() %>% 
  summary_plot(is_home)

# Model preprocessing -----------------------------------------------------

xgboost_recipe <- 
  recipe(formula = threat ~ ., data = kicks) %>% 
  step_novel(all_nominal_predictors()) %>% 
  # Tuned for threshold = 0.01, 0.005, 0.001
  step_other(all_nominal_predictors(), threshold = 0.001) %>% 
  step_unknown(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_zv(all_predictors()) 

# Check pre-process
xgboost_recipe %>%
  prep %>%
  juice

# Create test/train splits
train_split <- initial_split(kicks, strata = threat)
test <- testing(train_split)
train <- training(train_split)

# Create a small sample for tuning
train_cv <- 
  train %>% 
  sample_n(30000) %>% 
  rsample::vfold_cv(v = 5) 


xgboost_spec <- 
  boost_tree(
    trees = tune(),
    min_n = tune(),
    tree_depth = tune(),
    learn_rate = 0.02
  ) %>%
  set_mode("classification") %>%
  set_engine("xgboost") 

mset <- metric_set(roc_auc)

# Tune grid
xgb_grid <-
  grid_latin_hypercube(trees(c(400, 700)), 
                       min_n(c(15, 30)),
                       tree_depth(c(3, 8)),
                       size = 20)

xgboost_workflow <- 
  workflow() %>% 
  add_recipe(xgboost_recipe) %>% 
  add_model(xgboost_spec) 

#doParallel::registerDoParallel(cores = 5)

# Tune hyper-parameters
xgboost_tune <-
  tune_grid(
    xgboost_workflow,
    resamples = train_cv,
    grid = xgb_grid,
    control = control_resamples(verbose = T),
    metrics = mset
  )

xgboost_tune %>% autoplot() 

xgboost_tune %>% select_best()


# Check fit on holdout set
xgb_last_fit <- finalize_workflow(xgboost_workflow, tibble(trees = 550, min_n = 20, tree_depth = 4)) %>% 
  last_fit(train_split)

xgb_last_fit %>% collect_metrics()

# Fit final model
final_model <- finalize_workflow(xgboost_workflow, tibble(trees = 550, min_n = 20, tree_depth = 4)) %>% 
  fit(kicks)

# Feature importance ------------------------------------------------------

importances <- xgboost::xgb.importance(model = final_model$fit$fit$fit)

# It looks like distance from goal is more predictive than x/y position
importances %>%
  mutate(Feature = fct_reorder(Feature, Gain)) %>%
  ggplot(aes(Gain, Feature)) +
  geom_point()

# Formula trials ----------------------------------------------------------

xgboost_spec_tuned <- 
  boost_tree(
    trees = 550,
    min_n = 20,
    tree_depth = 4,
    learn_rate = 0.02
  ) %>%
  set_mode("classification") %>%
  set_engine("xgboost") 

formulas <- leave_var_out_formulas(formula = threat ~ ., data = kicks)

preproc <- map(formulas, function(f){
  recipe(formula = f, data = kicks) %>% 
    step_novel(all_nominal_predictors()) %>% 
    # Tuned for threshold = 0.01, 0.005, 0.001
    step_other(all_nominal_predictors(), threshold = 0.001) %>% 
    step_unknown(all_nominal_predictors()) %>% 
    step_dummy(all_nominal_predictors()) %>% 
    step_zv(all_predictors()) 
})

formula_trials <- 
  workflow_set(preproc = preproc,
               models = list(xgb = xgboost_spec_tuned)) %>%
  workflow_map("fit_resamples", resamples = train_cv, metrics = mset, control = control_resamples(verbose = T))

formula_trials %>% collect_metrics()


# Compute AUC loss --------------------------------------------------------

roc_values <- 
  formula_trials %>% 
  collect_metrics(summarize = FALSE) %>% 
  filter(.metric == "roc_auc") %>% 
  mutate(wflow_id = gsub("_xgb", "", wflow_id))

full_model <- 
  roc_values %>% 
  filter(wflow_id == "everything") %>% 
  select(full_model = .estimate, id)

differences <- 
  roc_values %>% 
  filter(wflow_id != "everything") %>% 
  full_join(full_model, by = "id") %>% 
  mutate(performance_drop = full_model - .estimate)

summary_stats <- 
  differences %>% 
  group_by(wflow_id) %>% 
  summarize(
    std_err = sd(performance_drop)/sum(!is.na(performance_drop)),
    performance_drop = mean(performance_drop),
    lower = performance_drop - qnorm(0.975) * std_err,
    upper = performance_drop + qnorm(0.975) * std_err,
    .groups = "drop"
  ) %>% 
  mutate(
    wflow_id = factor(wflow_id),
    wflow_id = reorder(wflow_id, performance_drop)
  )

summary_stats %>% filter(lower > 0)

ggplot(summary_stats, aes(x = performance_drop, y = wflow_id)) + 
  geom_point() + 
  geom_errorbar(aes(xmin = lower, xmax = upper), width = .25) +
  ylab("")



# Check effect of with/without distance ---------------------------------------

formulas <- list(full = threat ~ ., minus_dist = threat ~ desc_last + x_norm + y_norm + x_next1 + y_next1 + pass_angle + pass_distance + is_home)

preproc <- map(formulas, function(f){
  recipe(formula = f, data = kicks) %>% 
    step_novel(all_nominal_predictors()) %>% 
    # Tuned for threshold = 0.01, 0.005, 0.001
    step_other(all_nominal_predictors(), threshold = 0.001) %>% 
    step_unknown(all_nominal_predictors()) %>% 
    step_dummy(all_nominal_predictors()) %>% 
    step_zv(all_predictors()) 
})

formula_trials <- 
  workflow_set(preproc = preproc,
               models = list(xgb = xgboost_spec_tuned)) %>%
  workflow_map("fit_resamples", resamples = train_cv, metrics = mset, control = control_resamples(verbose = T))

formula_trials %>% collect_metrics()