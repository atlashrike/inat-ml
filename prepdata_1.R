library(rgbif)
library(purrr) 
library(data.table)
library(suncalc)
library(lubridate)
library(ggplot2)
library(dplyr)
library(arrow)

Sys.setenv(TZ = "UTC")

# get keys to download , eg
# aves_key <- name_backbone("Aves")$usageKey    

ds <- dataset_search(query = "iNaturalist research-grade observations")
inat_key <- ds$data$datasetKey[1]   

# this is to clean + get orders from a dataset w/ lit. labels
# some code missing here but you just need to get a list of orders otherwise 
# you just need to feed the order key to pred_in()
orders_clean <- unique(trimws(sub("/.*$", "", orders)))

safe_key <- possibly(
  function(ord) {
    hit <- name_backbone(name = ord, rank = "ORDER")
    if (nrow(hit) == 0 || is.null(hit$usageKey)) return(NA_real_)
    hit$usageKey[1]                # first row’s key
  },
  otherwise = NA_real_
)
order_keys <- orders_clean |>
  map_dbl(safe_key) |>
  na.omit() |>
  unique()  
failed <- setdiff(orders_clean, orders_clean[orders_clean %in% names(order_keys)])
if (length(failed))
  message("Orders not found in GBIF backbone: ",
          paste(failed, collapse = ", "))

dl <- occ_download(
  pred_and(
    pred_in("taxonKey", order_keys),
    pred("hasCoordinate", TRUE),
    pred("datasetKey", inat_key),
    pred_in("basisOfRecord",
            c("HUMAN_OBSERVATION", "OBSERVATION"))
  ),
  format = "SIMPLE_CSV",
  user   = "USER",
  pwd    = "PASSWORD",
  email  = "EMAIL"
)

# honestly redundant
dl_key <- as.character(dl)       

# status of download + download, because i like monitoring
repeat {
  status <- occ_download_meta(dl_key)$status
  message(format(Sys.time(), "%T"), " – ", status)
  if (status == "SUCCEEDED") break
  if (status == "KILLED") stop("GBIF cancelled the job")
  Sys.sleep(30)
}

# actual directing file to directory
dl_file <- occ_download_get(dl_key, overwrite = TRUE) 
zip_path <- dl_file                           
unzip(zip_path, exdir = "gbif_raw")   


#calculating solar stuff

calculate_solar_times <- function(dt) {
  dt[, lat_round := round(decimalLatitude, 1)]
  dt[, lon_round := round(decimalLongitude, 1)]
  dt[, date := as.Date(datetime_local)] #rounding the latitude and longitudes + handling date
  
  # calc sun times 4 location date pairs
  sun_times <- dt[, {
    times <- getSunlightTimes(
      date = date[1],
      lat = lat_round[1],
      lon = lon_round[1],
      keep = c("sunrise", "sunset", "dawn", "dusk", 
               "nauticalDawn", "nauticalDusk"),
      tz = "UTC"
    )
    .(
      nautical_dawn = times$nauticalDawn,
      civil_dawn = times$dawn,
      sunrise = times$sunrise,
      sunset = times$sunset,
      civil_dusk = times$dusk,
      nautical_dusk = times$nauticalDusk
    )
  }, by = .(lat_round, lon_round, date)]
  
  dt <- merge(dt, sun_times, 
              by = c("lat_round", "lon_round", "date"), 
              all.x = TRUE, sort = FALSE)
  
  return(dt)
}

classify_diel_period <- function(dt) {
  dt[, diel_period := fcase(
    datetime_local >= nautical_dusk | datetime_local < nautical_dawn, "night",
    datetime_local >= nautical_dawn & datetime_local < sunrise, "dawn",
    datetime_local >= sunrise & datetime_local < sunset, "day",
    datetime_local >= sunset & datetime_local < nautical_dusk, "dusk",
    default = NA_character_
  )]
  
  return(dt)
}

########(˶˃ ᵕ ˂˶)####
# trial wan: NORMALIZE WITHIN EACH PERIOD 
#  classify each observation -> day/night/dawn/dusk
#  calculate species frequency within each period 
#  each period sums to 1.0 independently
########(˶˃ ᵕ ˂˶)####

normalize_within_periods <- function(dt) {
  message("normalizing within each diel period... :>")
  period_totals <- dt[, .(total_obs = .N), by = diel_period]
  species_periods <- dt[, .(species_obs = .N), 
                        by = .(species, diel_period)]
  normalized <- merge(species_periods, period_totals, by = "diel_period")
  normalized[, within_period_freq := species_obs / total_obs]
  
  # gives the proportion of observations that belong to each species
  # within each period -> period independently sums to 1.0 across all species.
  
  # relative frequencies across periods for each species
  normalized[, total_species_obs := sum(species_obs), by = species]
  normalized[, raw_period_proportion := species_obs / total_species_obs]
  
  # effort correction by dividing by total observation rate
  effort_correction <- period_totals[, .(
    diel_period,
    effort_weight = total_obs / sum(total_obs)
  )]
  
  normalized <- merge(normalized, effort_correction, by = "diel_period")
  normalized[, corrected_activity := raw_period_proportion / effort_weight]
  
  # rescale so each species sums to 1.0 across periods
  normalized[, activity_proportion := corrected_activity / sum(corrected_activity), 
             by = species]
  
  return(normalized)
}

########(˶˃ ᵕ ˂˶)####
#  two: normalize across all periods
#  calculate total observations across all periods
#  calculate species proportion of total dataset
#  break down by period
########(˶˃ ᵕ ˂˶)####

normalize_across_periods <- function(dt) {
  total_all <- nrow(dt)
  
  # species-period combinations as proportion of total
  species_period_counts <- dt[, .(
    count = .N,
    proportion_of_total = .N / total_all
  ), by = .(species, diel_period)]
  
  # calculate effort correction based on period representation
  period_effort <- dt[, .(
    period_total = .N,
    period_proportion = .N / total_all
  ), by = diel_period]
  
  # merge and apply correction
  normalized <- merge(species_period_counts, period_effort, by = "diel_period")
  
  # correct for sampling effort 
  normalized[, effort_corrected := proportion_of_total / period_proportion]
  
  # rescale so each species sums to 1.0
  normalized[, activity_proportion := effort_corrected / sum(effort_corrected),
             by = species]
  
  return(normalized)
}

########(˶˃ ᵕ ˂˶)####
##  hourly normalization for both methods
########(˶˃ ᵕ ˂˶)####

normalize_hourly_within <- function(dt) {
  # first method applied to hourly bins
  dt[, hour := hour(datetime_local)]
  # calculate proportions within each hour
  hour_totals <- dt[, .(total_obs = .N), by = hour]
  species_hours <- dt[, .(species_obs = .N), by = .(species, hour)]
  
  normalized <- merge(species_hours, hour_totals, by = "hour")
  normalized[, within_hour_freq := species_obs / total_obs]
  normalized[, effort_weight := total_obs / sum(total_obs)]
  normalized[, corrected_activity := within_hour_freq / effort_weight]
  normalized[, activity_density := corrected_activity / sum(corrected_activity),
             by = species]
  
  return(normalized)
}

normalize_hourly_across <- function(dt) {
  # method 2 applied to hourly bins
  dt[, hour := hour(datetime_local)]
  
  total_all <- nrow(dt)
  
  # calculate proportions across entire dataset
  normalized <- dt[, .(
    count = .N,
    proportion_of_total = .N / total_all
  ), by = .(species, hour)]
  
  # hour effort
  hour_effort <- dt[, .(
    hour_total = .N,
    hour_proportion = .N / total_all
  ), by = hour]
  
  normalized <- merge(normalized, hour_effort, by = "hour")
  normalized[, effort_corrected := proportion_of_total / hour_proportion]
  normalized[, activity_density := effort_corrected / sum(effort_corrected),
             by = species]
  
  return(normalized)
}

########(˶˃ ᵕ ˂˶)####
##  the classifierrrrrrr
########(˶˃ ᵕ ˂˶)####

classify_activity_pattern <- function(activity_data) {
  # aggregate by species and period
  species_summary <- activity_data[, .(
    total_activity = sum(activity_proportion)
  ), by = .(species, diel_period)]

  species_wide <- dcast(species_summary, species ~ diel_period, 
                        value.var = "total_activity", fill = 0)
  
  # classify based on thresholds
  species_wide[, activity_pattern := fcase(
    day > 0.7, "diurnal",
    night > 0.7, "nocturnal",
    (dawn + dusk) > 0.5, "crepuscular",
    day > 0.4 & night > 0.3, "cathemeral",
    default = "cathemeral"
  )]
  
  # add confidence metric ie how concentrated is the activity
  species_wide[, concentration := pmax(day, night, dawn + dusk)]
  
  return(species_wide)
}

########(˶˃ ᵕ ˂˶)#### visualize

plot_normalization_comparison <- function(results_within, results_across, species_name) {
  within_data <- results_within[species == species_name]
  within_data[, method := "Within Periods"]
  
  across_data <- results_across[species == species_name]
  across_data[, method := "Across Periods"]
  
  combined <- rbind(
    within_data[, .(species, diel_period, activity_proportion, method)],
    across_data[, .(species, diel_period, activity_proportion, method)]
  )
  
  p <- ggplot(combined, aes(x = diel_period, y = activity_proportion, fill = method)) +
    geom_col(position = "dodge", alpha = 0.8) +
    scale_fill_manual(values = c("Within Periods" = "#3498db", 
                                 "Across Periods" = "#e74c3c")) +
    labs(title = paste("Normalization Method Comparison:", species_name),
         x = "Diel Period",
         y = "Activity Proportion",
         fill = "Normalization Method") +
    theme_minimal() +
    theme(legend.position = "top")
  
  return(p)
}

plot_hourly_comparison <- function(hourly_within, hourly_across, species_name) {
  within_data <- hourly_within[species == species_name]
  within_data[, method := "Within Hours"]
  
  across_data <- hourly_across[species == species_name]
  across_data[, method := "Across Hours"]
  
  combined <- rbind(
    within_data[, .(species, hour, activity_density, method)],
    across_data[, .(species, hour, activity_density, method)]
  )
  
  p <- ggplot(combined, aes(x = hour, y = activity_density, color = method)) +
    geom_line(size = 1.2) +
    geom_point(size = 2) +
    scale_color_manual(values = c("Within Hours" = "#3498db", 
                                  "Across Hours" = "#e74c3c")) +
    scale_x_continuous(breaks = seq(0, 23, 2)) +
    labs(title = paste("Hourly Activity -", species_name),
         x = "Hour of Day",
         y = "Activity Density",
         color = "Normalization Method") +
    theme_minimal() +
    theme(legend.position = "top")
  
  return(p)
}

########(˶˃ ᵕ ˂˶)#### analysis

analyze_activity_both_methods <- function(file_paths, 
                                          method = c("both", "within", "across"),
                                          min_obs = 50) {
  
  method <- match.arg(method)

  message("Loading data...")
  cols <- c("eventDate", "decimalLatitude", "decimalLongitude", 
            "species", "class", "order", "family")
  
  dt <- rbindlist(
    lapply(file_paths, fread, select = cols, showProgress = TRUE),
    use.names = TRUE, fill = TRUE
  )

  dt <- dt[!is.na(eventDate) & !is.na(decimalLatitude) & !is.na(decimalLongitude)]
  dt[, datetime_utc := parse_date_time(
    eventDate,
    orders = c("ymd_HMS", "ymd_HM", "ymd", "Ymd_HMSz"),
    tz = "UTC",
    quiet = TRUE
  )]

  dt <- dt[!is.na(datetime_utc) & 
             (hour(datetime_utc) + minute(datetime_utc) + second(datetime_utc)) != 0]

  dt[, datetime_local := datetime_utc + dhours(decimalLongitude / 15)]

  message("Calculating solar positions...")
  dt <- calculate_solar_times(dt)
  dt <- classify_diel_period(dt)
  
  species_counts <- dt[, .N, by = species]
  good_species <- species_counts[N >= min_obs, species]
  dt <- dt[species %in% good_species]
  
  message(paste("Analyzing", uniqueN(dt$species), "species with >=", min_obs, "observations"))
  results <- list()
  if (method %in% c("both", "within")) {
    message("\nApplying Method 1: Within-period normalization...")
    results$within_periods <- normalize_within_periods(dt)
    results$within_hourly <- normalize_hourly_within(dt)
    results$within_classifications <- classify_activity_pattern(results$within_periods)
  }
  
  if (method %in% c("both", "across")) {
    message("\nApplying Method 2: Across-period normalization...")
    results$across_periods <- normalize_across_periods(dt)
    results$across_hourly <- normalize_hourly_across(dt)
    results$across_classifications <- classify_activity_pattern(results$across_periods)
  }
  
  results$raw_data <- dt

  if (method == "both") {
    message("\nComparing classification results...")
    comparison <- merge(
      results$within_classifications[, .(species, within_pattern = activity_pattern, 
                                         within_confidence = concentration)],
      results$across_classifications[, .(species, across_pattern = activity_pattern,
                                         across_confidence = concentration)],
      by = "species"
    )
    comparison[, agreement := within_pattern == across_pattern]
    results$method_comparison <- comparison

    agreement_rate <- comparison[, mean(agreement)]
    message(sprintf("Classification agreement between methods: %.1f%%", 
                    agreement_rate * 100))
  }
  
  return(results)
}

########(˶˃ ᵕ ˂˶)#### diagnostic

diagnose_normalization_effects <- function(results) {
  raw_counts <- results$raw_data[, .(
    raw_count = .N
  ), by = .(species, diel_period)]

  if (!is.null(results$within_periods)) {
    within_norm <- results$within_periods[, .(
      species, diel_period, 
      within_normalized = activity_proportion
    )]
    raw_counts <- merge(raw_counts, within_norm, by = c("species", "diel_period"))
  }
  
  if (!is.null(results$across_periods)) {
    across_norm <- results$across_periods[, .(
      species, diel_period,
      across_normalized = activity_proportion
    )]
    raw_counts <- merge(raw_counts, across_norm, by = c("species", "diel_period"))
  }

  raw_counts[, raw_proportion := raw_count / sum(raw_count), by = species]
  
  return(raw_counts)
}

########(˶˃ ᵕ ˂˶)#### testing
files <- c("gbif_raw/insects.csv")

results <- analyze_activity_both_methods(
  files, 
  method = "both",  # "both", "within", "across"
  min_obs = 50
)

example_species <- "Antrostomus vociferus"

p1 <- plot_normalization_comparison(
  results$within_periods,
  results$across_periods,
  example_species
)
ggsave("normalization_comparison_periods.png", p1, width = 10, height = 6)

p2 <- plot_hourly_comparison(
  results$within_hourly,
  results$across_hourly,
  example_species
)
ggsave("normalization_comparison_hourly.png", p2, width = 10, height = 6)

if (!is.null(results$method_comparison)) {
  disagreements <- results$method_comparison[agreement == FALSE]
  print("Species with different classifications between methods:")
  print(disagreements)
}

diagnostic <- diagnose_normalization_effects(results)
print(diagnostic[species == example_species])

if (!is.null(results$within_classifications)) {
  fwrite(results$within_classifications, "classifications_within_method.csv")
}
if (!is.null(results$across_classifications)) {
  fwrite(results$across_classifications, "classifications_across_method.csv")
}

########(˶˃ ᵕ ˂˶)#### summ report

generate_comparison_report <- function(results) {
  cat("\n=== NORMALIZATION METHOD COMPARISON REPORT ===\n\n")
  
  if (!is.null(results$within_classifications)) {
    cat("METHOD 1 (Within Periods) Results:\n")
    within_summary <- results$within_classifications[, .N, by = activity_pattern]
    print(within_summary)
    cat("\n")
  }
  
  if (!is.null(results$across_classifications)) {
    cat("METHOD 2 (Across Periods) Results:\n")
    across_summary <- results$across_classifications[, .N, by = activity_pattern]
    print(across_summary)
    cat("\n")
  }
  
  if (!is.null(results$method_comparison)) {
    cat("Method Agreement:\n")
    cat(sprintf("- Total species analyzed: %d\n", nrow(results$method_comparison)))
    cat(sprintf("- Species with same classification: %d (%.1f%%)\n",
                sum(results$method_comparison$agreement),
                mean(results$method_comparison$agreement) * 100))
    
    cat("\nSpecies with highest confidence (Method 1):\n")
    top_within <- results$within_classifications[order(-concentration)][1:10]
    print(top_within[, .(species, activity_pattern, confidence = round(concentration, 3))])
    
    cat("\nSpecies with largest method disagreement:\n")
    disagreements <- results$method_comparison[agreement == FALSE]
    disagreements[, confidence_diff := abs(within_confidence - across_confidence)]
    print(disagreements[order(-confidence_diff)][1:10, 
                                                 .(species, within_pattern, across_pattern, confidence_diff = round(confidence_diff, 3))])
  }
}

message("Script loaded successfully!")
message("Main function: analyze_activity_both_methods()")
message("  - method = 'both': Run both normalization methods")
message("  - method = 'within': Only within-period normalization")
message("  - method = 'across': Only across-period normalization")



library(arrow)

save_dir <- "results"
dir.create(save_dir, showWarnings = FALSE)

write_parquet(results$raw_data, file.path(save_dir, "diel_data_bug.parquet"), compression = "snappy")

if (!is.null(results$within_periods))
  write_parquet(results$within_periods, file.path(save_dir, "within_periods_bug.parquet"), compression = "snappy")

if (!is.null(results$across_periods))
  write_parquet(results$across_periods, file.path(save_dir, "across_periods_bug.parquet"), compression = "snappy")

if (!is.null(results$within_hourly))
  write_parquet(results$within_hourly, file.path(save_dir, "within_hourly_bug.parquet"), compression = "snappy")

if (!is.null(results$across_hourly))
  write_parquet(results$across_hourly, file.path(save_dir, "across_hourly_bug.parquet"), compression = "snappy")

if (!is.null(results$within_classifications))
  write_parquet(results$within_classifications, file.path(save_dir, "within_classifications_bug.parquet"), compression = "snappy")

if (!is.null(results$across_classifications))
  write_parquet(results$across_classifications, file.path(save_dir, "across_classifications_bug.parquet"), compression = "snappy")

if (!is.null(results$method_comparison))
  write_parquet(results$method_comparison, file.path(save_dir, "method_comparison_bug.parquet"), compression = "snappy")


save_dir <- "results"
dir.create(save_dir, showWarnings = FALSE)

save_csv <- function(x, fname, ...) {
  fwrite(x,
         file = file.path(save_dir, fname),
         quote = TRUE,         
         sep   = ",",
         na    = "",           
         ...)
  message("✓ wrote ", file.path(save_dir, fname))
}

save_csv(results$raw_data, "diel_data.csv")

save_csv(results$within_periods,         "within_periods_fishe.csv")
save_csv(results$across_periods,         "across_periods_fishe.csv")
save_csv(results$within_hourly,          "within_hourly_fishe.csv")
save_csv(results$across_hourly,          "across_hourly_fishe.csv")
save_csv(results$within_classifications, "within_classifications_fishe.csv")
save_csv(results$across_classifications, "across_classifications_fishe.csv")
save_csv(results$method_comparison,      "method_comparison_fishe.csv")

