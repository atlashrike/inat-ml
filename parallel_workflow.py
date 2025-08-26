#!/usr/bin/env python3
# parallel_workflow.py - parallelized helpers (solar-aware merge)

import os
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib


def extract_features_single(species_data_tuple):
    species, species_data = species_data_tuple

    hourly_values = np.zeros(24, dtype=float)
    for h in range(24):
        hour_data = species_data[species_data['hour'] == h]
        if len(hour_data) > 0:
            hourly_values[h] = float(hour_data['activity_density'].values[0])

    # normalize to a distribution
    total = hourly_values.sum()
    if total > 0:
        hourly_values = hourly_values / total

    features = {'species': species}
    for h in range(24):
        features[f'hour_{h}'] = hourly_values[h]

    night_hours = np.array([20, 21, 22, 23, 0, 1, 2, 3, 4, 5])
    day_hours = np.arange(6, 18)
    dawn_hours = np.array([4, 5, 6, 7])
    dusk_hours = np.array([17, 18, 19, 20])

    features['night_activity'] = hourly_values[night_hours].sum()
    features['day_activity'] = hourly_values[day_hours].sum()
    features['dawn_activity'] = hourly_values[dawn_hours].sum()
    features['dusk_activity'] = hourly_values[dusk_hours].sum()

    features['peak_hour'] = int(np.argmax(hourly_values))
    features['peak_value'] = float(hourly_values.max())
    features['activity_variance'] = float(np.var(hourly_values))
    features['activity_concentration'] = float(hourly_values.max() / (hourly_values.sum() + 1e-10))
    features['night_day_ratio'] = float(features['night_activity'] / (features['day_activity'] + 1e-10))
    features['twilight_ratio'] = float(features['dawn_activity'] + features['dusk_activity'])

    return features


def extract_features_parallel(hourly_data, n_jobs=-1):
    """Parallel feature extraction (clock-based, merged with solar later if provided)."""
    if n_jobs == -1:
        n_jobs = cpu_count()
    print(f"extracting features using {n_jobs} cores...")

    species_groups = [(species, group) for species, group in hourly_data.groupby('species')]
    with Pool(processes=n_jobs) as pool:
        features_list = pool.map(extract_features_single, species_groups)
    return pd.DataFrame(features_list)


def train_model_single(config):
    model_type, X_train, y_train, X_test, y_test = config

    if model_type == 'logistic':
        model = LogisticRegression(multi_class='multinomial', max_iter=1000, class_weight='balanced')
    elif model_type == 'random_forest':
        model = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight='balanced', n_jobs=-1, random_state=42)
    else:
        return None

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    accuracy = model.score(X_test, y_test)

    return {'model_type': model_type, 'model': model, 'accuracy': accuracy, 'train_time': train_time}


def train_models_parallel(X_train, y_train, X_test, y_test):
    model_types = ['logistic', 'random_forest']
    configs = [(m, X_train, y_train, X_test, y_test) for m in model_types]

    print(f"training {len(model_types)} models in parallel...")
    results = {}
    with ProcessPoolExecutor(max_workers=len(model_types)) as executor:
        futures = {executor.submit(train_model_single, cfg): cfg[0] for cfg in configs}
        for fut in as_completed(futures):
            m = futures[fut]
            try:
                r = fut.result()
                results[m] = r
                print(f"  {m}: accuracy={r['accuracy']:.3f}, time={r['train_time']:.1f}s")
            except Exception as e:
                print(f"  {m} failed: {e}")
    return results


def run_parallel_analysis(data_dir="ml_data", output_dir="ml_results"):
    print("parallel ml analysis")
    print(f"using {cpu_count()} cpu cores")

    Path(output_dir).mkdir(exist_ok=True)

    # load hourly & labels
    hourly_data = pd.read_csv(f"{data_dir}/hourly_activity_within.csv")
    labels = pd.read_csv(f"{data_dir}/literature_labels.csv")

    print(f"  {len(hourly_data)} observations")
    print(f"  {hourly_data['species'].nunique()} species")

    # parallel feature extraction
    print("\nextracting features...")
    start = time.time()
    features = extract_features_parallel(hourly_data)
    print(f"  done in {time.time() - start:.1f}s")

    # load and merge solar periods if available
    pfile = os.path.join(data_dir, "period_features.csv")
    if os.path.exists(pfile):
        solar = pd.read_csv(pfile)
        cols = ['night_activity', 'day_activity', 'dawn_activity', 'dusk_activity']
        s = solar[cols].sum(axis=1).replace(0, np.nan)
        for c in cols:
            solar[c] = solar[c] / s
        solar = solar.fillna(0)
        features = features.merge(solar, on='species', how='left', suffixes=('', '_solar'))
        for col in cols:
            sc = col + '_solar'
            if sc in features:
                features[col] = features[sc].where(~features[sc].isna(), features[col])
                features.drop(columns=[sc], inplace=True, errors='ignore')
        features['night_day_ratio'] = (features['night_activity'] + 1e-6) / (features['day_activity'] + 1e-6)
        print(f"  merged solar periods for {solar['species'].nunique()} species")

    # prepare training data
    train_data = features.merge(labels[['species', 'pattern']], on='species').dropna(subset=['pattern'])
    if len(train_data) < 20:
        print("not enough training data")
        return None

    X = train_data.drop(['species', 'pattern'], axis=1)
    y = train_data['pattern']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train in parallel
    start = time.time()
    model_results = train_models_parallel(X_train_scaled, y_train, X_test_scaled, y_test)
    total_time = time.time() - start
    print(f"\ntotal training time: {total_time:.1f}s")

    best_model_type = max(model_results, key=lambda k: model_results[k]['accuracy'])
    best_result = model_results[best_model_type]
    print(f"\nbest model: {best_model_type}  |  accuracy: {best_result['accuracy']:.3f}")

    # predict all species
    X_all = features.drop(['species'], axis=1)
    X_all_scaled = scaler.transform(X_all)
    preds = best_result['model'].predict(X_all_scaled)

    results_df = pd.DataFrame({'species': features['species'], 'predicted_pattern': preds})
    results_df.to_csv(f"{output_dir}/predictions_parallel.csv", index=False)
    joblib.dump(best_result['model'], f"{output_dir}/best_model_parallel.pkl")

    print(f"\nresults saved to {output_dir}/")
    return results_df


if __name__ == "__main__":
    _ = run_parallel_analysis()
