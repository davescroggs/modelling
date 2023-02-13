library(tidyverse)
library(tidymodels)

pumpkins_raw <- readr::read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-10-19/pumpkins.csv") %>% 
  mutate(across(place,as.integer),
         across(c(weight_lbs,ott,est_weight,pct_chart),as.double)) %>% 
  separate(id,c("year","type"),sep = "-")


# Create test/train sets --------------------------------------------------

pumpkins_split <- pumpkins_raw %>% initial_split()

pumpkins_test <- pumpkins_split %>% testing()
pumpkins_train <- pumpkins_split %>% training()

pumkins_folds <- pumpkins_train %>% vfold_cv(v = 5)



# Set up engines ----------------------------------------------------------

pump_xgb <- parsnip::boost_tree(mode = "regression",
                                mtry = tune(),
                                trees = tune(),
                                min_n = tune(),
                                learn_rate = 0.02)


# Pre-processing ----------------------------------------------------------

# Country matters
# Grower matters
# Year might matter
# Type might matter
# State matters


pump_rec <- recipe(weight_lbs ~ ., data = pumpkins_train)
