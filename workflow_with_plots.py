#!/usr/bin/env python3
# workflow_with_plots.py - complete workflow with comprehensive visualizations (solar-aware + NaN cleaning)

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ML models
from ml_models import (
    ActivityPatternClassifier,
    SelfSupervisedActivityLearner,
    EnsembleActivityClassifier,
    prepare_data_from_r,
)

# Parallel helpers
from parallel_workflow import (
    extract_features_parallel,
    train_models_parallel
)

# Visualizations (assume these exist in your repo)
from visualizations import (
    plot_activity_patterns_by_class,
    plot_detailed_confusion_matrix,
    plot_feature_importance_comparison,
    plot_confidence_analysis,
    plot_species_curves,
    plot_temporal_distributions,
    plot_model_comparison,
    plot_misclassification_analysis,
    plot_peak_hour_analysis,
    plot_bimodality_analysis,
    plot_learning_curves,
    plot_roc_curves,
    plot_activity_heatmap,
    create_summary_report
)


# ---------- NEW: robust cleaning helper ----------
def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replace inf/-inf with NaN, then fill NaN with 0. Only numeric columns."""
    df = df.copy()
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) == 0:
        return df
    df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
    df[num_cols] = df[num_cols].fillna(0.0)
    return df
# -------------------------------------------------


def _load_solar_periods_table(data_dir: str):
    """
    Loads optional R-exported solar period features:
    expected columns: species, dawn_activity, day_activity, dusk_activity, night_activity
    """
    pfile = os.path.join(data_dir, "period_features.csv")
    if os.path.exists(pfile):
        df = pd.read_csv(pfile)
        needed = {'species', 'night_activity', 'day_activity', 'dawn_activity', 'dusk_activity'}
        if not needed.issubset(set(df.columns)):
            print(f"[warn] {pfile} missing columns {needed - set(df.columns)}; ignoring.")
            return None
        # normalize to [0,1] proportions if not already
        cols = ['night_activity', 'day_activity', 'dawn_activity', 'dusk_activity']
        s = df[cols].sum(axis=1)
        s = s.replace(0, np.nan)
        for c in cols:
            df[c] = df[c] / s
        df = df.fillna(0)
        print(f"loaded solar period table with {len(df)} species from {pfile}")
        return df
    print("no solar period table found (period_features.csv) – using legacy fixed windows")
    return None


def _inject_solar_periods(features: pd.DataFrame, solar_periods: pd.DataFrame) -> pd.DataFrame:
    """
    Merge solar-aware period columns into the features, preferring solar values when present.
    """
    if solar_periods is None or features is None or features.empty:
        return features

    merged = features.merge(solar_periods, on='species', how='left', suffixes=('', '_solar'))
    for col in ['night_activity', 'day_activity', 'dawn_activity', 'dusk_activity']:
        solar = col + '_solar'
        if solar in merged.columns:
            merged[col] = merged[solar].where(~merged[solar].isna(), merged[col])
            merged.drop(columns=[solar], inplace=True, errors='ignore')

    # keep ratio consistent
    if all(c in merged.columns for c in ['night_activity', 'day_activity']):
        merged['night_day_ratio'] = (merged['night_activity'] + 1e-6) / (merged['day_activity'] + 1e-6)

    return merged


def run_complete_analysis_with_plots(
    data_dir="ml_data",
    output_dir="ml_results",
    use_parallel=True,
    create_plots=True,
    n_plot_examples=20
):
    """complete ml workflow with all visualizations (solar-aware)"""

    print("temporal activity ml analysis with visualizations")
    print("=" * 60)

    # setup
    Path(output_dir).mkdir(exist_ok=True)
    start_time = time.time()

    # 1) load data exported from R
    print("\n1. loading data...")
    hourly_data, lit_labels = prepare_data_from_r(
        f"{data_dir}/hourly_activity_within.csv",
        f"{data_dir}/literature_labels.csv"
    )
    print(f"  loaded {len(hourly_data)} hourly rows across {hourly_data['species'].nunique()} species")
    print(f"  literature labels: {len(lit_labels)} species")

    # optional: species metadata
    try:
        metadata = pd.read_csv(f"{data_dir}/species_metadata.csv")
    except Exception:
        metadata = None

    # load solar period table (if present)
    solar_periods = _load_solar_periods_table(data_dir)

    # 2) feature extraction
    print("\n2. extracting features...")
    if use_parallel:
        print("  using parallel extraction...")
        features = extract_features_parallel(hourly_data)
        # inject solar-aware period proportions
        features = _inject_solar_periods(features, solar_periods)
    else:
        clf = ActivityPatternClassifier()
        features = clf.prepare_features(hourly_data, solar_periods=solar_periods)

    # ---------- NEW: clean features to prevent NaNs/Infs downstream ----------
    features = _clean_features(features)
    # ------------------------------------------------------------------------

    print(f"  extracted {len(features.columns)} features for {len(features)} species")

    # 3) create training strategies (literature, self, mixed)
    print("\n3. creating training sets...")

    lit_train = lit_labels[lit_labels['confidence'].fillna(0) >= 3].dropna(subset=['pattern']).copy()
    print(f"  literature: {len(lit_train)} high-confidence species")

    # self-supervised (now benefits from solar-aware periods)
    ssl = SelfSupervisedActivityLearner()
    self_train = ssl.create_reference_patterns(hourly_data, min_confidence=0.85, solar_periods=solar_periods)
    print(f"  self-supervised: {len(self_train)} patterns identified")

    # mixed strategy = literature strong + solar-SSL remainder
    mixed_train = lit_train.copy()
    new_species = self_train[~self_train['species'].isin(lit_train['species'])]
    mixed_train = pd.concat([lit_train, new_species], ignore_index=True).dropna(subset=['pattern'])
    print(f"  mixed: {len(mixed_train)} total species")

    # save training sets
    Path(output_dir).mkdir(exist_ok=True)
    lit_train.to_csv(f"{output_dir}/training_literature.csv", index=False)
    self_train.to_csv(f"{output_dir}/training_self_supervised.csv", index=False)
    mixed_train.to_csv(f"{output_dir}/training_mixed.csv", index=False)

    # 4) prepare training data
    print("\n4. preparing training data (mixed)...")
    train_features = features[features['species'].isin(mixed_train['species'])]
    train_merged = train_features.merge(mixed_train[['species', 'pattern']], on='species')

    if len(train_merged) < 20:
        print("  error: not enough training data")
        return None

    X = train_merged.drop(['species', 'pattern'], axis=1)
    y = train_merged['pattern']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"  training set: {len(X_train)}  |  test set: {len(X_test)}")
    print(f"  classes: {sorted(y.unique())}")

    # 5) train models
    print("\n5. training models...")
    if use_parallel:
        model_results = train_models_parallel(X_train_scaled, y_train, X_test_scaled, y_test)
    else:
        model_results = {}
        # logistic regression
        print("  training logistic regression...")
        lr = ActivityPatternClassifier()
        lr.train(train_merged.drop('pattern', axis=1), y, model_type='logistic')
        model_results['logistic'] = {
            'model': lr.model,
            'accuracy': lr.model.score(X_test_scaled, y_test),
            'cv_score': 0.0, 'cv_std': 0.0, 'train_time': 0.0
        }
        # random forest
        print("  training random forest...")
        rf = ActivityPatternClassifier()
        rf.train(train_merged.drop('pattern', axis=1), y, model_type='random_forest')
        model_results['random_forest'] = {
            'model': rf.model,
            'accuracy': rf.model.score(X_test_scaled, y_test),
            'cv_score': 0.0, 'cv_std': 0.0, 'train_time': 0.0
        }

    # 6) select best model
    print("\n6. selecting best model...")
    best_model_type = max(model_results, key=lambda k: model_results[k]['accuracy'])
    best_result = model_results[best_model_type]
    print(f"  best model: {best_model_type} (accuracy={best_result['accuracy']:.3f})")

    # persist
    joblib.dump(best_result['model'], f"{output_dir}/best_model.pkl")
    joblib.dump(scaler, f"{output_dir}/scaler.pkl")

    # 7) predict all species
    print("\n7. predicting all species...")
    X_all = features.drop(['species'], axis=1)
    X_all_scaled = scaler.transform(X_all)
    predictions = best_result['model'].predict(X_all_scaled)

    results_df = pd.DataFrame({'species': features['species'], 'predicted_pattern': predictions})
    if hasattr(best_result['model'], 'predict_proba'):
        proba = best_result['model'].predict_proba(X_all_scaled)
        for i, cls in enumerate(best_result['model'].classes_):
            results_df[f'prob_{cls}'] = proba[:, i]
        results_df['confidence'] = proba.max(axis=1)

    # attach literature labels if present
    results_df = results_df.merge(lit_labels[['species', 'pattern']], on='species', how='left')
    # attach metadata if present
    if metadata is not None:
        results_df = results_df.merge(metadata, on='species', how='left')

    results_df.to_csv(f"{output_dir}/all_predictions.csv", index=False)
    print(f"  predicted patterns for {len(results_df)} species")

    # 8) visualizations
    if create_plots:
        print("\n8. creating comprehensive visualizations...")
        plot_activity_patterns_by_class(hourly_data, results_df, output_dir)
        if 'pattern' in results_df.columns:
            plot_detailed_confusion_matrix(results_df, output_dir)
        plot_feature_importance_comparison(model_results, output_dir)
        plot_confidence_analysis(results_df, output_dir)
        plot_species_curves(hourly_data, results_df, output_dir, n_examples=n_plot_examples)
        plot_temporal_distributions(hourly_data, results_df, output_dir)
        plot_model_comparison(model_results, output_dir)
        if 'pattern' in results_df.columns:
            plot_misclassification_analysis(results_df, hourly_data, output_dir)
        plot_peak_hour_analysis(hourly_data, results_df, output_dir)
        plot_bimodality_analysis(hourly_data, results_df, output_dir)
        try:
            plot_learning_curves(model_results, X_train_scaled, y_train, output_dir)
        except Exception:
            print("  learning curves skipped")
        try:
            plot_roc_curves(model_results, X_test_scaled, y_test, output_dir)
        except Exception:
            print("  ROC curves skipped")
        plot_activity_heatmap(hourly_data, results_df, output_dir)
        create_summary_report(results_df, model_results, output_dir)
        print(f"\n  all visualizations saved to {output_dir}/")

    # 9) summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"time elapsed: {elapsed:.1f} s")
    print(f"best model: {best_model_type}  |  accuracy: {best_result['accuracy']:.3f}")
    if 'predicted_pattern' in results_df.columns:
        counts = results_df['predicted_pattern'].value_counts()
        for k, v in counts.items():
            print(f"  {k}: {v} ({v / len(results_df) * 100:.1f}%)")
    if 'confidence' in results_df.columns:
        print(f"mean confidence: {results_df['confidence'].mean():.3f}")
        print(f">0.7: {(results_df['confidence'] > 0.7).sum()} species")
        print(f">0.9: {(results_df['confidence'] > 0.9).sum()} species")

    print("\nfiles saved:")
    print(f"  - {output_dir}/all_predictions.csv")
    print(f"  - {output_dir}/best_model.pkl")
    if create_plots:
        print(f"  - {output_dir}/*.png")
        print(f"  - {output_dir}/summary_report.txt")

    return {
        'predictions': results_df,
        'model_results': model_results,
        'best_model': (best_model_type, best_result),
        'features': features,
        'hourly_data': hourly_data
    }


def run_quick_analysis(data_dir="ml_data", output_dir="ml_results"):
    return run_complete_analysis_with_plots(
        data_dir=data_dir,
        output_dir=output_dir,
        use_parallel=True,
        create_plots=True,
        n_plot_examples=12
    )


if __name__ == "__main__":
    results = run_quick_analysis()
