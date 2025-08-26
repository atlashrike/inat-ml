library(data.table)
library(reticulate)
library(dplyr)       
library(lubridate)  
 
canon_pattern <- function(x) {
  x <- tolower(trimws(as.character(x)))
  x[x %in% c("diurnal","day","day-active","day active")] <- "diurnal"
  x[x %in% c("nocturnal","night","night-active","night active")] <- "nocturnal"
  x[x %in% c("crepuscular","twilight","dawn/dusk","dawn-dusk","dawn dusk")] <- "crepuscular"
  x[x %in% c("cathemeral","arrhythmic","irregular")] <- "cathemeral"
  x[!(x %in% c("diurnal","nocturnal","crepuscular","cathemeral"))] <- NA_character_
  x
}


clean_lit_data_v2 <- function(path_or_dt, min_conf = 3L, prefer_main = TRUE,
                              conf_reduce = c("max","mean")) {
  conf_reduce <- match.arg(conf_reduce)
  
  lit <- if (is.character(path_or_dt)) data.table::fread(path_or_dt, encoding = "UTF-8")
  else data.table::as.data.table(path_or_dt)
  data.table::setDT(lit)
  
  # normalize names -> snake_case
  data.table::setnames(lit, names(lit), gsub("[^A-Za-z0-9]+", "_", names(lit)))
  
  req <- c("Species_name","Diel_Pattern")
  if (!all(req %in% names(lit))) stop("Expected columns 'Species_name' and 'Diel_Pattern' not found.")
  
  # confidence columns
  conf_cols <- grep("^Confidence_[1-4]$", names(lit), value = TRUE)
  if (!length(conf_cols) && "Confidence" %in% names(lit)) conf_cols <- "Confidence"
  for (cn in conf_cols) data.table::set(lit, j = cn, value = suppressWarnings(as.numeric(lit[[cn]])))
  
  # main/alt canonical labels
  data.table::set(lit, j = "main_pattern", value = canon_pattern(lit[["Diel_Pattern"]]))
  if ("AltDiel_Pattern" %in% names(lit)) {
    data.table::set(lit, j = "alt_pattern", value = canon_pattern(lit[["AltDiel_Pattern"]]))
  } else {
    data.table::set(lit, j = "alt_pattern", value = rep(NA_character_, nrow(lit)))
  }
  
  # collapse confidence
  if (length(conf_cols)) {
    conf_vec <- if (conf_reduce == "max") {
      if (length(conf_cols) == 1L) lit[[conf_cols]]
      else do.call(pmax, c(lit[, ..conf_cols], list(na.rm = TRUE)))
    } else {
      rowMeans(lit[, ..conf_cols], na.rm = TRUE)
    }
    data.table::set(lit, j = "confidence", value = conf_vec)
  } else {
    data.table::set(lit, j = "confidence", value = rep(NA_real_, nrow(lit)))
  }
  
  # row-wise pattern preference (vectorized)
  pat <- if (prefer_main) {
    data.table::fifelse(!is.na(lit$main_pattern), lit$main_pattern, lit$alt_pattern)
  } else {
    data.table::fifelse(!is.na(lit$alt_pattern), lit$alt_pattern, lit$main_pattern)
  }
  
  species <- gsub("_", " ", trimws(lit$Species_name))
  out <- data.table::data.table(species = species, pattern = pat, confidence = lit$confidence)
  
  # filter & dedupe
  out <- out[!is.na(pattern) & (is.na(confidence) | confidence >= min_conf)]
  data.table::setorder(out, species, -confidence)
  out <- out[!duplicated(species)]
  out[]
}


ensure_species_col <- function(dt) {
  setDT(dt)
  if (!"species" %in% names(dt)) {
    cand <- intersect(c("Species_name","species","scientificName","canonicalName","taxonName"), names(dt))
    if (length(cand) == 0) stop("No species-like column found in data.")
    set(dt, j = "species", value = trimws(gsub("_"," ", as.character(dt[[cand[1]]]))))
  } else {
    dt[, species := trimws(gsub("_"," ", species))]
  }
  dt[]
}

prepare_ml_data <- function(results, lit_data, output_dir = "ml_data", min_conf = 3L) {
  lit_labels   <- if (is.character(lit_data)) clean_lit_data_v2(lit_data, min_conf = min_conf)
  else clean_lit_data_v2(lit_data, min_conf = min_conf)
  
  hourly_within <- ensure_species_col(as.data.table(results$within_hourly))
  hourly_across <- ensure_species_col(as.data.table(results$across_hourly))
  raw_data      <- ensure_species_col(as.data.table(results$raw_data))
  dir.create(output_dir, showWarnings = FALSE)
  
  # ensure consistent format
  hourly_within[, method := "within"]
  hourly_across[, method := "across"]
  
  # combine if needed
  all_hourly <- rbindlist(list(hourly_within, hourly_across), use.names = TRUE, fill = TRUE)
  
  # save hourly data
  fwrite(hourly_within, file.path(output_dir, "hourly_activity_within.csv"))
  fwrite(hourly_across, file.path(output_dir, "hourly_activity_across.csv"))
  fwrite(all_hourly,    file.path(output_dir, "hourly_activity_combined.csv"))
  
  # save cleaned literature labels
  fwrite(lit_labels, file.path(output_dir, "literature_labels.csv"))
  
  # raw observation metadata per species
  obs_counts <- raw_data[, .(
    n_obs   = .N,
    n_hours = uniqueN(lubridate::hour(datetime_local)),
    lat_range = max(decimalLatitude,  na.rm = TRUE) - min(decimalLatitude,  na.rm = TRUE),
    lon_range = max(decimalLongitude, na.rm = TRUE) - min(decimalLongitude, na.rm = TRUE)
  ), by = species]
  fwrite(obs_counts, file.path(output_dir, "species_metadata.csv"))
  
  cat("exported data to", output_dir, "\n",
      "files created:\n",
      "- hourly_activity_within.csv\n",
      "- hourly_activity_across.csv\n",
      "- hourly_activity_combined.csv\n",
      "- literature_labels.csv\n",
      "- species_metadata.csv\n", sep = "")
}

create_ml_training_sets <- function(results, lit_data, strategy = "literature",
                                    min_conf = 3L, strict_conf = 4L) {
  if (strategy == "literature") {
    cat("using literature labels for training (new schema)…\n")
    labels <- if (is.character(lit_data)) clean_lit_data_v2(lit_data, min_conf = min_conf)
    else clean_lit_data_v2(lit_data, min_conf = min_conf)
    
  } else if (strategy == "self_identified") {
    cat("identifying clear patterns from data…\n")
    activity_summary <- results$within_periods[, .(total_activity = sum(activity_proportion)),
                                               by = .(species, diel_period)]
    species_wide <- dcast(activity_summary, species ~ diel_period,
                          value.var = "total_activity", fill = 0)
    labels <- species_wide[, .(
      species = species,
      pattern = dplyr::case_when(
        day   > 0.85 ~ "diurnal",
        night > 0.85 ~ "nocturnal",
        (dawn + dusk) > 0.70 ~ "crepuscular",
        TRUE ~ NA_character_
      ),
      confidence = pmax(day, night, dawn + dusk, na.rm = TRUE)
    )][!is.na(pattern)]
    
  } else if (strategy == "mixed") {
    cat("using mixed strategy…\n")
    lit_labels <- if (is.character(lit_data)) clean_lit_data_v2(lit_data, min_conf = min_conf)
    else clean_lit_data_v2(lit_data, min_conf = min_conf)
    
    activity_summary <- results$within_periods[, .(total_activity = sum(activity_proportion)),
                                               by = .(species, diel_period)]
    species_wide <- dcast(activity_summary, species ~ diel_period,
                          value.var = "total_activity", fill = 0)
    
    self_labels <- species_wide[, .(
      species = species,
      pattern = dplyr::case_when(
        day   > 0.90 ~ "diurnal",
        night > 0.90 ~ "nocturnal",
        (dawn + dusk) > 0.75 ~ "crepuscular",
        TRUE ~ NA_character_
      ),
      confidence = pmax(day, night, dawn + dusk, na.rm = TRUE)
    )][!is.na(pattern)]
    
    # prefer strong literature labels; fill with high-confidence self labels
    lit_strong <- lit_labels[confidence >= strict_conf]
    labels <- rbindlist(list(
      lit_strong,
      self_labels[!species %in% lit_strong$species]
    ), use.names = TRUE, fill = TRUE)
  }
  
  cat("created training set with", nrow(labels), "species\n")
  cat("pattern distribution:\n"); print(labels[, .N, by = pattern])
  labels[]
}

balance_training_data <- function(labels, hourly_data, balance_method = "downsample") {
  labels <- as.data.table(labels)
  pattern_counts <- labels[, .N, by = pattern]
  
  if (balance_method == "downsample") {
    min_n <- min(pattern_counts$N)
    balanced_labels <- labels[, .SD[sample(.N, min(min_n, .N))], by = pattern]
    
  } else if (balance_method == "upsample") {
    max_n <- max(pattern_counts$N)
    balanced_labels <- labels[, {
      n_needed <- max_n
      if (.N < n_needed) .SD[sample(.N, n_needed, replace = TRUE)] else .SD
    }, by = pattern]
    
  } else if (balance_method == "smote") {
    cat("smote balancing not implemented in R, use python\n")
    balanced_labels <- labels
  }
  
  cat("\nbalanced dataset:\n"); print(balanced_labels[, .N, by = pattern])
  balanced_labels[]
}

visualize_training_data <- function(hourly_data, labels, n_examples = 16) {
  if (!requireNamespace("ggplot2", quietly = TRUE) ||
      !requireNamespace("gridExtra", quietly = TRUE)) {
    stop("ggplot2 and gridExtra are required for plotting.")
  }
  
  example_species <- labels[, .SD[sample(.N, min(4, .N))], by = pattern]
  plots <- list()
  
  for (i in 1:nrow(example_species)) {
    sp <- example_species$species[i]
    pattern <- example_species$pattern[i]
    sp_data <- hourly_data[species == sp]
    
    if (nrow(sp_data) > 0) {
      p <- ggplot2::ggplot(sp_data, ggplot2::aes(x = hour, y = activity_density)) +
        ggplot2::geom_line(size = 1, color = "darkblue") +
        ggplot2::geom_point(size = 2, color = "darkblue") +
        ggplot2::geom_area(alpha = 0.3, fill = "lightblue") +
        ggplot2::scale_x_continuous(breaks = seq(0, 23, 6)) +
        ggplot2::labs(title = paste0(sp, "\n", pattern),
                      x = "hour", y = "activity") +
        ggplot2::theme_minimal() +
        ggplot2::theme(plot.title = ggplot2::element_text(size = 10))
      plots[[length(plots) + 1]] <- p
    }
  }
  
  gridExtra::grid.arrange(grobs = plots, ncol = 4)
}

run_ml_in_r <- function(hourly_data, labels) {
  if (!requireNamespace("mlr3", quietly = TRUE) ||
      !requireNamespace("mlr3learners", quietly = TRUE)) {
    cat("mlr3 not available, export data for python analysis\n"); return(NULL)
  }
  library(mlr3); library(mlr3learners)
  
  features <- prepare_features_r(hourly_data)
  ml_data  <- merge(features, labels[, .(species, pattern)], by = "species")
  ml_data  <- ml_data[!is.na(pattern)]
  
  task <- as_task_classif(ml_data, target = "pattern", id = "activity_patterns")
  
  learners <- list(
    lrn("classif.log_reg", predict_type = "prob"),
    lrn("classif.ranger",  predict_type = "prob", num.trees = 100),
    lrn("classif.xgboost", predict_type = "prob", nrounds = 100)
  )
  
  results <- list()
  for (learner in learners) {
    cat("\ntraining", learner$id, "...\n")
    rr <- resample(task, learner, rsmp("cv", folds = 5))
    results[[learner$id]] <- rr
    cat("accuracy:", round(rr$aggregate(msr("classif.acc")), 3), "\n")
  }
  results
}

# build features from sun-based periods (from results$within_periods)
build_period_features <- function(results) {
  wp <- data.table::as.data.table(results$within_periods)
  # wp has: species, diel_period, activity_proportion
  pf <- data.table::dcast(wp, species ~ diel_period,
                          value.var = "activity_proportion", fill = 0)
  # rename to match your old feature names
  data.table::setnames(pf,
                       old = c("dawn","day","dusk","night"),
                       new = c("dawn_activity","day_activity","dusk_activity","night_activity"),
                       skip_absent = TRUE
  )
  # a couple of composites
  pf[, crepuscularity := dawn_activity + dusk_activity]
  pf[, day_night_ratio := day_activity / pmax(night_activity, 1e-9)]
  pf[]
}
prepare_features_r <- function(hourly_data, results = NULL) {
  hd <- data.table::as.data.table(hourly_data)
  hd[, hour := as.integer(hour)]
  
  # keep purely hourly shape features (these don't assume clock=solar)
  features <- hd[, {
    hourly_vals <- numeric(24)
    for (h in 0:23) {
      v <- .SD[hour == h]$activity_density
      hourly_vals[h + 1] <- ifelse(length(v), v[1], 0)
    }
    list(
      peak_hour        = which.max(hourly_vals) - 1,
      activity_variance= var(hourly_vals),
      max_activity     = max(hourly_vals),
      n_peaks          = count_peaks(hourly_vals)
    )
  }, by = species]
  
  # add the 24 hourly columns (optional but useful)
  hourly_wide <- data.table::dcast(hd, species ~ hour,
                                   value.var = "activity_density", fill = 0)
  data.table::setnames(hourly_wide, as.character(0:23), paste0("hour_", 0:23))
  features <- merge(features, hourly_wide, by = "species", all.x = TRUE)
  
  # >>> sun-based period features (preferred over fixed hour windows)
  if (!is.null(results)) {
    pf <- build_period_features(results)
    features <- merge(features, pf, by = "species", all.x = TRUE)
  }
  
  features[]
}


# example usage:
# prepare_ml_data(results, "literature_new.csv", output_dir = "ml_data", min_conf = 3)
# lit_labels   <- create_ml_training_sets(results, "literature_new.csv", strategy = "literature", min_conf = 3)
# self_labels  <- create_ml_training_sets(results, "literature_new.csv", strategy = "self_identified")
# mixed_labels <- create_ml_training_sets(results, "literature_new.csv", strategy = "mixed",
#                                         min_conf = 3, strict_conf = 4)

