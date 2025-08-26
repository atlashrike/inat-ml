#!/usr/bin/env python3
# ml_models.py - core ml models for activity pattern classification (solar-aware refactor)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import warnings

warnings.filterwarnings('ignore')


class ActivityPatternClassifier:
    """
    Feature builder + thin wrappers around a few sklearn classifiers.

    Refactor highlights:
    - prepare_features(..., solar_periods=None) optionally consumes a per-species table
      with columns: ['species','night_activity','day_activity','dawn_activity','dusk_activity'].
      If provided, these overwrite the fixed-window derived features.
    - Backward compatible: if solar_periods is None (or species missing), falls back
      to the original fixed 06:00–18:00 windows.
    """

    def __init__(self, method='logistic'):
        self.method = method
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None

    def prepare_features(self, hourly_data, include_derived=True, solar_periods=None, normalize_hours=False):
        """
        Build per-species features from 24h activity curves.

        Parameters
        ----------
        hourly_data : DataFrame with columns ['species','hour','activity_density']
        include_derived : bool
            Whether to add peak/hourly-shape-derived features.
        solar_periods : DataFrame or None
            Optional per-species table with columns:
            ['species','night_activity','day_activity','dawn_activity','dusk_activity'].
            If provided, these values will be used instead of fixed-hour windows.
        normalize_hours : bool
            If True, normalize 24h vector to sum=1 before computing derived features.
        """
        features = []

        # pre-index optional solar_periods for O(1) lookups
        solar_idx = None
        if solar_periods is not None and len(solar_periods) > 0:
            need = {'species', 'night_activity', 'day_activity', 'dawn_activity', 'dusk_activity'}
            if not need.issubset(set(solar_periods.columns)):
                missing = need - set(solar_periods.columns)
                raise ValueError(f"solar_periods is missing required columns: {missing}")
            solar_idx = solar_periods.set_index('species')

        species_list = hourly_data['species'].astype(str).unique()
        for species in species_list:
            species_data = hourly_data[hourly_data['species'] == species].sort_values('hour')

            # 24h vector (0..23), with zeros for missing hours
            hourly_values = []
            for h in range(24):
                v = species_data.loc[species_data['hour'] == h, 'activity_density']
                hourly_values.append(float(v.values[0]) if len(v) else 0.0)

            hourly_array = np.array(hourly_values, dtype=float)
            if normalize_hours:
                s = hourly_array.sum()
                if s > 0:
                    hourly_array = hourly_array / s

            feature_dict = {f'hour_{h}': hourly_array[h] for h in range(24)}
            feature_dict['species'] = species

            if include_derived:
                # shape features
                feature_dict['peak_hour'] = int(np.argmax(hourly_array))
                feature_dict['activity_concentration'] = float(
                    hourly_array.max() / (hourly_array.sum() + 1e-6)
                )
                feature_dict['activity_variance'] = float(np.var(hourly_array))
                feature_dict['n_peaks'] = int(self._count_peaks(hourly_array))

                # period summaries (solar-aware if provided)
                if solar_idx is not None and species in solar_idx.index:
                    row = solar_idx.loc[species]
                    feature_dict['night_activity'] = float(row['night_activity'])
                    feature_dict['day_activity']   = float(row['day_activity'])
                    feature_dict['dawn_activity']  = float(row['dawn_activity'])
                    feature_dict['dusk_activity']  = float(row['dusk_activity'])
                else:
                    # legacy fixed windows (clock-based)
                    night_hours = [20, 21, 22, 23, 0, 1, 2, 3, 4, 5]
                    day_hours   = list(range(6, 18))
                    dawn_hours  = [4, 5, 6, 7]
                    dusk_hours  = [17, 18, 19, 20]
                    feature_dict['night_activity'] = float(hourly_array[night_hours].sum())
                    feature_dict['day_activity']   = float(hourly_array[day_hours].sum())
                    feature_dict['dawn_activity']  = float(hourly_array[dawn_hours].sum())
                    feature_dict['dusk_activity']  = float(hourly_array[dusk_hours].sum())

                feature_dict['night_day_ratio'] = (
                    (feature_dict['night_activity'] + 1e-6) / (feature_dict['day_activity'] + 1e-6)
                )

            features.append(feature_dict)

        return pd.DataFrame(features)

    def _count_peaks(self, activity_array):
        smoothed = np.convolve(activity_array, np.ones(3) / 3, mode='same')
        peaks = 0
        for i in range(1, len(smoothed) - 1):
            if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]:
                if smoothed[i] > 0.1 * np.max(smoothed):
                    peaks += 1
        return peaks

    def train(self, features_df, labels, model_type='logistic'):
        X = features_df.drop(['species'], axis=1)
        self.feature_names = X.columns.tolist()

        X_scaled = self.scaler.fit_transform(X)

        if model_type == 'logistic':
            self.model = LogisticRegression(
                multi_class='multinomial',
                max_iter=1000,
                class_weight='balanced'
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boost':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                random_state=42
            )
        elif model_type == 'neural_net':
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=800,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model.fit(X_scaled, labels)

        # cross validation (quick)
        try:
            cv_scores = cross_val_score(self.model, X_scaled, labels, cv=5, n_jobs=-1)
            print(f"cv accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        except Exception as e:
            print(f"cv skipped: {e}")

        return self

    def predict(self, features_df):
        species_col = features_df['species']
        X = features_df.drop(['species'], axis=1)
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        results = pd.DataFrame({'species': species_col, 'prediction': predictions})

        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
            for i, class_name in enumerate(self.model.classes_):
                results[f'prob_{class_name}'] = probabilities[:, i]
            results['confidence'] = probabilities.max(axis=1)

        return results

    def feature_importance(self):
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)

        elif hasattr(self.model, 'coef_'):
            importances = np.mean(np.abs(self.model.coef_), axis=0)
            feature_imp = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
        else:
            feature_imp = None

        return feature_imp


class SelfSupervisedActivityLearner:
    """
    Produces high-confidence reference species directly from features.
    Uses solar-aware period features automatically if they exist in the features_df.
    """

    def __init__(self):
        self.reference_patterns = {}
        self.classifier = None

    def create_reference_patterns(self, hourly_data, min_confidence=0.8, solar_periods=None):
        # Build features (solar-aware if provided)
        features_df = ActivityPatternClassifier().prepare_features(
            hourly_data, include_derived=True, solar_periods=solar_periods
        )

        reference_species = []
        for _, row in features_df.iterrows():
            species = row['species']

            night = row.get('night_activity', 0.0)
            day   = row.get('day_activity', 0.0)
            dawn  = row.get('dawn_activity', 0.0)
            dusk  = row.get('dusk_activity', 0.0)

            confidence = 0.0
            pattern = None

            if (night / (night + day + 1e-6)) > 0.8:
                pattern = 'nocturnal'
                confidence = night / (night + day + 1e-6)
            elif (day / (night + day + 1e-6)) > 0.8:
                pattern = 'diurnal'
                confidence = day / (night + day + 1e-6)
            elif ((dawn + dusk) / (night + day + dawn + dusk + 1e-6)) > 0.6:
                pattern = 'crepuscular'
                confidence = (dawn + dusk) / (night + day + dawn + dusk + 1e-6)

            if pattern and confidence >= min_confidence:
                reference_species.append({
                    'species': species,
                    'pattern': pattern,
                    'confidence': float(confidence)
                })

        reference_df = pd.DataFrame(reference_species)
        print(f"found {len(reference_df)} high-confidence reference species")
        if not reference_df.empty:
            print(reference_df['pattern'].value_counts())

        # store mean patterns (hourly curves) for each class
        for pattern in reference_df['pattern'].unique():
            pattern_species = reference_df[reference_df['pattern'] == pattern]['species'].tolist()
            # take the hourly rows for those species from hourly_data
            pattern_data = hourly_data[hourly_data['species'].isin(pattern_species)]

            mean_pattern = []
            for h in range(24):
                hour_activities = pattern_data[pattern_data['hour'] == h]['activity_density'].values
                mean_pattern.append(np.mean(hour_activities) if len(hour_activities) > 0 else 0.0)

            self.reference_patterns[pattern] = np.array(mean_pattern)

        return reference_df

    def train_from_references(self, hourly_data, reference_df, solar_periods=None):
        all_features = ActivityPatternClassifier().prepare_features(
            hourly_data, include_derived=True, solar_periods=solar_periods
        )
        train_features = all_features[all_features['species'].isin(reference_df['species'])]
        train_data = train_features.merge(reference_df[['species', 'pattern']], on='species')

        self.classifier = ActivityPatternClassifier()
        self.classifier.train(train_features.drop(columns=[]), train_data['pattern'], model_type='random_forest')
        return self.classifier


class CurveSimilarityClassifier:
    def __init__(self):
        self.reference_curves = {}
        self.weights = None

    def set_reference_curves(self, reference_patterns):
        self.reference_curves = reference_patterns

    def calculate_similarity(self, activity_curve, reference_curve, method='correlation'):
        if method == 'correlation':
            similarity = np.corrcoef(activity_curve, reference_curve)[0, 1]
        elif method == 'cosine':
            similarity = np.dot(activity_curve, reference_curve) / (
                np.linalg.norm(activity_curve) * np.linalg.norm(reference_curve) + 1e-6
            )
        elif method == 'weighted_euclidean':
            # legacy: emphasize night and edge hours
            weights = np.ones(24)
            weights[[20, 21, 22, 23, 0, 1, 2, 3, 4, 5]] = 2.0
            weights[[6, 7, 17, 18, 19]] = 1.5
            distance = np.sqrt(np.sum(weights * (activity_curve - reference_curve) ** 2))
            similarity = 1 / (1 + distance)
        else:
            raise ValueError(f"Unknown method: {method}")
        return similarity

    def classify(self, hourly_data, method='correlation'):
        results = []
        for species in hourly_data['species'].unique():
            sd = hourly_data[hourly_data['species'] == species].sort_values('hour')
            curve = np.array([sd.loc[sd['hour'] == h, 'activity_density'].values[0] if (sd['hour'] == h).any() else 0.0
                              for h in range(24)])

            sims = {p: self.calculate_similarity(curve, ref, method) for p, ref in self.reference_curves.items()}
            best = max(sims, key=sims.get)
            results.append({'species': species, 'prediction': best, **{f'sim_{k}': v for k, v in sims.items()}})

        return pd.DataFrame(results)


class EnsembleActivityClassifier:
    def __init__(self):
        self.models = {}
        self.weights = {}

    def train_ensemble(self, hourly_data, labels_df, include_self_supervised=True, solar_periods=None):
        feature_prep = ActivityPatternClassifier()
        features = feature_prep.prepare_features(hourly_data, solar_periods=solar_periods)

        train_data = features.merge(labels_df[['species', 'pattern']], on='species')
        train_data = train_data.dropna(subset=['pattern'])
        X = train_data.drop(['species', 'pattern'], axis=1)
        y = train_data['pattern']

        model_types = ['logistic', 'random_forest', 'gradient_boost']
        for model_type in model_types:
            print(f"training {model_type}...")
            clf = ActivityPatternClassifier()
            clf.train(train_data.drop('pattern', axis=1), y, model_type=model_type)
            self.models[model_type] = clf

        if include_self_supervised:
            print("adding self-supervised model...")
            ssl = SelfSupervisedActivityLearner()
            reference_df = ssl.create_reference_patterns(hourly_data, min_confidence=0.85, solar_periods=solar_periods)
            ssl.train_from_references(hourly_data, reference_df, solar_periods=solar_periods)
            self.models['self_supervised'] = ssl.classifier

        # curve similarity model
        print("adding curve similarity...")
        curve_clf = CurveSimilarityClassifier()
        reference_patterns = {}
        for pattern in labels_df['pattern'].unique():
            species_pool = labels_df[labels_df['pattern'] == pattern]['species']
            sample = species_pool.sample(min(10, len(species_pool)), random_state=42) if len(species_pool) else species_pool
            pattern_data = hourly_data[hourly_data['species'].isin(sample)]

            mean_curve = [np.mean(pattern_data[pattern_data['hour'] == h]['activity_density'].values)
                          if len(pattern_data[pattern_data['hour'] == h]) > 0 else 0.0
                          for h in range(24)]
            reference_patterns[pattern] = np.array(mean_curve)

        curve_clf.set_reference_curves(reference_patterns)
        self.models['curve_similarity'] = curve_clf

        # equal weights for now
        n_models = len(self.models)
        for model_name in self.models:
            self.weights[model_name] = 1.0 / n_models
        print(f"model weights: {self.weights}")

        return self

    def predict_ensemble(self, hourly_data, solar_periods=None):
        features = ActivityPatternClassifier().prepare_features(hourly_data, solar_periods=solar_periods)

        all_predictions = {}
        for model_name, model in self.models.items():
            print(f"getting predictions from {model_name}...")
            try:
                if model_name == 'curve_similarity':
                    preds = model.classify(hourly_data, method='weighted_euclidean')
                else:
                    preds = model.predict(features)
                if preds is None or (isinstance(preds, pd.DataFrame) and preds.empty):
                    print(f"  warning: {model_name} returned no predictions")
                    continue
                all_predictions[model_name] = preds
            except Exception as e:
                print(f"  error in {model_name}: {str(e)}")
                continue

        if not all_predictions:
            print("  no models produced predictions")
            return pd.DataFrame()

        species_list = features['species'].unique()
        ensemble_results = []

        for species in species_list:
            votes = {}
            for model_name, df in all_predictions.items():
                row = df[df['species'] == species]
                if row.empty:
                    continue
                if 'prediction' in row.columns:
                    pred_pattern = row.iloc[0]['prediction']
                elif 'pattern' in row.columns:
                    pred_pattern = row.iloc[0]['pattern']
                else:
                    continue
                votes[pred_pattern] = votes.get(pred_pattern, 0.0) + self.weights.get(model_name, 1.0)

            if not votes:
                ensemble_results.append({'species': species, 'ensemble_prediction': 'unknown', 'ensemble_confidence': 0.0})
                continue

            final_pred = max(votes, key=votes.get)
            total_weight = sum(votes.values())
            confidence = votes[final_pred] / total_weight if total_weight > 0 else 0.0
            ensemble_results.append({'species': species, 'ensemble_prediction': final_pred, 'ensemble_confidence': confidence})

        return pd.DataFrame(ensemble_results)


def prepare_data_from_r(hourly_csv_path, labels_csv_path):
    """
    Robust loader & column normalizer for R-exported CSVs.
    """
    import re

    hourly = pd.read_csv(hourly_csv_path)
    labels = pd.read_csv(labels_csv_path)

    def norm_cols(df):
        df = df.copy()
        df.columns = [re.sub(r'[^0-9a-zA-Z]+', '_', c).strip('_').lower() for c in df.columns]
        return df

    hourly = norm_cols(hourly)
    labels = norm_cols(labels)

    def ensure_species(df):
        cand = [c for c in ['species', 'species_name', 'scientificname', 'canonicalname', 'taxonname', 'speciesname']
                if c in df.columns]
        if not cand:
            raise ValueError(f"no species column found. columns: {list(df.columns)}")
        if 'species' not in df.columns:
            df = df.rename(columns={cand[0]: 'species'})
        df['species'] = df['species'].astype(str).str.replace('_', ' ').str.strip()
        return df

    hourly = ensure_species(hourly)
    labels = ensure_species(labels)

    # hour column
    if 'hour' not in hourly.columns:
        for alt in ['solar_hour', 'bin', 'hour_bin', 'h']:
            if alt in hourly.columns:
                hourly = hourly.rename(columns={alt: 'hour'})
                break
    if 'hour' not in hourly.columns:
        raise ValueError(f"hour column missing. columns: {list(hourly.columns)}")
    hourly['hour'] = hourly['hour'].astype(int)

    # activity column
    if 'activity_density' not in hourly.columns:
        for alt in ['activity_proportion', 'activity', 'density', 'value']:
            if alt in hourly.columns:
                hourly = hourly.rename(columns={alt: 'activity_density'})
                break
    if 'activity_density' not in hourly.columns:
        raise ValueError(f"activity_density column missing. columns: {list(hourly.columns)}")

    # pattern/diel column
    label_col = next((c for c in ['pattern', 'diel', 'diel_pattern', 'dielpattern'] if c in labels.columns), None)
    if label_col is None:
        raise ValueError(f"no label column found. columns: {list(labels.columns)}")

    def canon(x):
        x = str(x).strip().lower()
        if x in {'diurnal', 'day', 'day-active', 'day active'}: return 'diurnal'
        if x in {'nocturnal', 'night', 'night-active', 'night active'}: return 'nocturnal'
        if x in {'crepuscular', 'twilight', 'dawn/dusk', 'dawn-dusk', 'dawn dusk'}: return 'crepuscular'
        if x in {'cathemeral', 'arrhythmic', 'irregular'}: return 'cathemeral'
        return None

    labels['pattern'] = labels[label_col].map(canon)

    # confidence if present
    import numpy as _np
    conf_cols = [c for c in labels.columns if c == 'confidence' or re.fullmatch(r'confidence_\d+', c)]
    if conf_cols:
        conf_df = labels[conf_cols].apply(pd.to_numeric, errors='coerce')
        labels['confidence'] = conf_df.max(axis=1)
    elif 'confidence' not in labels.columns:
        labels['confidence'] = _np.nan

    labels = labels[['species', 'pattern', 'confidence']].dropna(subset=['pattern']).drop_duplicates()
    hourly = hourly[['species', 'hour', 'activity_density']]

    return hourly, labels


if __name__ == "__main__":
    print("ml models loaded - solar-aware refactor ready")
