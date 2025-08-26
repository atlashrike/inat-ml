library(data.table)
library(arrow)
library(ggplot2)
library(stringr)

# recreate the results object by loading the saved files
results <- list()

# load from parquet files
results$raw_data <- read_parquet("results/diel_data_bug.parquet")
results$within_periods <- read_parquet("results/within_periods_bug.parquet")
results$across_periods <- read_parquet("results/across_periods_bug.parquet")
results$within_hourly <- read_parquet("results/within_hourly_bug.parquet")
results$across_hourly <- read_parquet("results/across_hourly_bug.parquet")
results$within_classifications <- read_parquet("results/within_classifications_bug.parquet")
results$across_classifications <- read_parquet("results/across_classifications_bug.parquet")
results$method_comparison <- read_parquet("results/method_comparison_buge.parquet")
lit_data <- readRDS("trait_data_fish.rds")
lit_data <- fread("insectsraw.csv",
            na.strings = c("", "NA"),
            encoding = "UTF-8")

library(data.table)


source("ml2scripts2.R")

prepare_ml_data(results, lit_data, output_dir = "ml_data")

# create different training strategies
lit_labels <- create_ml_training_sets(results, lit_data, strategy = "literature")
self_labels <- create_ml_training_sets(results, lit_data, strategy = "self_identified")
mixed_labels <- create_ml_training_sets(results, lit_data, strategy = "mixed")

# visualize training examples to check they make sense
library(gridExtra)
viz <- visualize_training_data(results$within_hourly, mixed_labels)
ggsave("training_examples.png", viz, width = 16, height = 12)
