#!/usr/bin/env python3
# visualizations.py — publication-friendly plots (solar-aware, robust to sparse hours)

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Utilities & global style
# =========================

def _ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def _set_style():
    # Neutral, poster-friendly defaults
    sns.set_theme(context="talk", style="whitegrid")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.titleweight": "bold",
        "axes.titlesize": 16,
        "axes.labelweight": "bold",
        "axes.labelsize": 13,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
    })

def _reindex_24(series: pd.Series) -> pd.Series:
    """Ensure hour-indexed Series has indices 0..23; fill missing with 0."""
    return series.reindex(range(24), fill_value=0)

def _sum_hours(series: pd.Series, hours) -> float:
    """Sum over the provided hour list safely (handles missing hours)."""
    return _reindex_24(series).reindex(hours, fill_value=0).sum()

def _load_solar_periods_table(data_dir: str):
    """
    Load optional R-exported solar period features:
    expected columns: species, dawn_activity, day_activity, dusk_activity, night_activity
    Returns normalized per-species proportions or None if missing.
    """
    pfile = os.path.join(data_dir, "period_features.csv")
    if not os.path.exists(pfile):
        return None
    df = pd.read_csv(pfile)
    need = {'species', 'night_activity', 'day_activity', 'dawn_activity', 'dusk_activity'}
    if not need.issubset(df.columns):
        return None
    cols = ['night_activity', 'day_activity', 'dawn_activity', 'dusk_activity']
    s = df[cols].sum(axis=1).replace(0, np.nan)
    for c in cols:
        df[c] = df[c] / s
    df = df.fillna(0)
    return df

# Consistent palettes / labels
CLASSES = ['nocturnal', 'diurnal', 'crepuscular', 'cathemeral']
PRETTY  = {'nocturnal':'Nocturnal','diurnal':'Diurnal',
           'crepuscular':'Crepuscular','cathemeral':'Cathemeral'}
COLORS  = {'nocturnal':'#2c3e50','diurnal':'#f39c12',
           'crepuscular':'#8e44ad','cathemeral':'#27ae60'}



# =========================================
# 1) Activity patterns by predicted class
# =========================================
def plot_activity_patterns_by_class(hourly_data, predictions, output_dir):
    """plot mean activity curves for each predicted class"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()

    patterns = ['nocturnal', 'diurnal']
    colors = {'nocturnal': '#2c3e50', 'diurnal': '#f39c12'}

    for idx, pattern in enumerate(patterns):
        ax = axes[idx]

        # get species for this pattern
        if 'predicted_pattern' in predictions.columns:
            pattern_species = predictions[predictions['predicted_pattern'] == pattern]['species']
        elif 'ensemble_prediction' in predictions.columns:
            pattern_species = predictions[predictions['ensemble_prediction'] == pattern]['species']
        else:
            pattern_species = predictions[predictions['prediction'] == pattern]['species']

        if len(pattern_species) == 0:
            ax.text(0.5, 0.5, f'no {pattern} species', ha='center', va='center')
            ax.set_title(f'{pattern.capitalize()} Pattern')
            continue

        # get hourly data for these species
        pattern_data = hourly_data[hourly_data['species'].isin(pattern_species)]

        # calculate mean and std for each hour
        hourly_stats = pattern_data.groupby('hour')['activity_density'].agg(['mean', 'std', 'sem'])

        hours = hourly_stats.index
        means = hourly_stats['mean'].values
        sems = hourly_stats['sem'].values

        # plot mean with confidence interval
        ax.plot(hours, means, color=colors[pattern], linewidth=2, label='mean')
        ax.fill_between(hours, means - sems, means + sems,
                        color=colors[pattern], alpha=0.3, label='±SEM')

        # add individual species as thin lines
        for species in pattern_species[:10]:  # show first 10
            species_data = hourly_data[hourly_data['species'] == species]
            species_hourly = species_data.groupby('hour')['activity_density'].mean()
            ax.plot(species_hourly.index, species_hourly.values,
                    color=colors[pattern], alpha=0.1, linewidth=0.5)

        # day/night shading & cosmetics
        ax.axvspan(0, 6,  alpha=0.05, color='blue',  label='night')
        ax.axvspan(6, 18, alpha=0.05, color='yellow', label='day')
        ax.axvspan(18, 24, alpha=0.05, color='blue')

        ax.set_title(f'{pattern.capitalize()} Pattern (n={len(pattern_species)})')
        ax.set_xlabel('hour of day')
        ax.set_ylabel('activity density')
        ax.set_xlim(0, 23)
        ax.set_xticks(range(0, 24, 3))
        ax.grid(True, alpha=0.3)

        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Activity Patterns by Classification', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/activity_patterns_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()




# =========================================
# 2) Confusion matrix (counts & percents)
# =========================================
def plot_detailed_confusion_matrix(results_df, output_dir):
    _set_style(); _ensure_dir(output_dir)
    if not {'pattern', 'predicted_pattern'}.issubset(results_df.columns):
        print("confusion matrix skipped (requires columns: 'pattern', 'predicted_pattern')")
        return

    y_true = results_df['pattern'].str.lower()
    y_pred = results_df['predicted_pattern'].str.lower()

    # Limit to known classes to avoid surprise categories
    y_true = y_true.where(y_true.isin(CLASSES), other=np.nan)
    y_pred = y_pred.where(y_pred.isin(CLASSES), other=np.nan)
    df = pd.DataFrame({'true': y_true, 'pred': y_pred}).dropna()
    if df.empty:
        print("confusion matrix skipped (no overlap after cleaning)")
        return

    cm = pd.crosstab(df['true'], df['pred'], dropna=False).reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    cm_pct = (cm.T / cm.sum(axis=1).replace(0, np.nan)).T.fillna(0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[PRETTY[c] for c in CLASSES],
                yticklabels=[PRETTY[c] for c in CLASSES],
                cbar_kws={'label': 'Count'}, ax=axes[0], square=True)
    axes[0].set_title('Confusion Matrix — Counts')
    axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True')

    sns.heatmap(cm_pct, annot=True, fmt='.0%', cmap='RdYlGn',
                xticklabels=[PRETTY[c] for c in CLASSES],
                yticklabels=[PRETTY[c] for c in CLASSES],
                vmin=0, vmax=1, cbar_kws={'label': 'Row %'}, ax=axes[1], square=True)
    axes[1].set_title('Confusion Matrix — Percent')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True')

    out = f"{output_dir}/confusion_matrix_detailed.png"
    plt.tight_layout(); plt.savefig(out, facecolor='white'); plt.close()
    print(f"confusion matrix saved to {out}")



# =========================================
# 3) Feature importance comparison
# =========================================
def plot_feature_importance_comparison(model_results, output_dir, top_n=20):
    _set_style(); _ensure_dir(output_dir)

    def _get_importances(m):
        if hasattr(m, "feature_importances_"):
            names = getattr(m, "feature_names_in_", None)
            if names is None:
                return None
            return pd.DataFrame({"feature": names, "importance": m.feature_importances_})
        if hasattr(m, "coef_"):
            names = getattr(m, "feature_names_in_", None)
            if names is None:
                return None
            imp = np.mean(np.abs(m.coef_), axis=0)
            return pd.DataFrame({"feature": names, "importance": imp})
        return None

    rows = []
    for key, obj in model_results.items():
        model = obj.get('model', None)
        if model is None:
            continue
        imp_df = _get_importances(model)
        if imp_df is None:
            continue
        imp_df = imp_df.sort_values("importance", ascending=False).head(top_n)
        imp_df["model"] = key
        rows.append(imp_df)

    if not rows:
        print("feature importance skipped (models lack feature names/importances)")
        return

    imp_all = pd.concat(rows, ignore_index=True)
    g = sns.catplot(
        data=imp_all, y="feature", x="importance", hue="model",
        kind="bar", height=8, aspect=1.2, legend_out=False
    )
    g.set_ylabels(""); g.set_xlabels("Importance"); g.fig.suptitle("Top Feature Importances", y=0.98)
    out = f"{output_dir}/feature_importance_comparison.png"
    plt.savefig(out, facecolor='white'); plt.close()
    print(f"feature importance saved to {out}")



# =========================================
# 4) Confidence analysis
# =========================================
def plot_confidence_analysis(results_df, output_dir):
    _set_style(); _ensure_dir(output_dir)
    if 'confidence' not in results_df.columns:
        print("confidence analysis skipped (no 'confidence' column)")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(results_df['confidence'], bins=30, ax=ax)
    ax.axvline(results_df['confidence'].mean(), ls='--', color='red',
               label=f"Mean = {results_df['confidence'].mean():.2f}")
    ax.set_title("Prediction Confidence Distribution"); ax.set_xlabel("Confidence"); ax.set_ylabel("Count")
    ax.legend()

    out = f"{output_dir}/confidence_distribution.png"
    plt.tight_layout(); plt.savefig(out, facecolor='white'); plt.close()
    print(f"confidence analysis saved to {out}")



# =========================================
# 5) Species curves (n examples)
# =========================================
def plot_species_curves(hourly_data, results_df, output_dir, n_examples=20):
    _set_style(); _ensure_dir(output_dir)
    pred_col = 'predicted_pattern' if 'predicted_pattern' in results_df.columns else 'prediction'
    # pick examples evenly by class
    examples = (results_df.groupby(pred_col, group_keys=False)
                          .apply(lambda d: d.sample(min(max(1, n_examples // 4), len(d)), random_state=42))
                          .reset_index(drop=True))
    if examples.empty:
        print("species curves skipped (no examples)"); return

    n = len(examples); ncols = 4; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 2.8*nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, row) in zip(axes, examples.iterrows()):
        sp = row['species']; cls = row[pred_col]
        sp_data = hourly_data[hourly_data['species'] == sp]
        curve = sp_data.groupby('hour')['activity_density'].mean()
        curve = _reindex_24(curve)
        ax.plot(np.arange(24), curve.values, lw=2, color=COLORS.get(cls, '#555'))
        ax.set_title(f"{sp}\n{PRETTY.get(cls, cls)}", fontsize=10)
        ax.set_xticks([0,6,12,18,23]); ax.grid(alpha=0.2)

    # hide any unused axes
    for ax in axes[n:]: ax.axis('off')

    plt.suptitle("Example Species Activity Curves", y=0.98, weight='bold')
    out = f"{output_dir}/species_curves.png"
    plt.tight_layout(); plt.savefig(out, facecolor='white'); plt.close()
    print(f"species curves saved to {out}")



# =========================================
# 6) Temporal distributions (SOLAR AWARE)
# =========================================
def plot_temporal_distributions(hourly_data, results_df, output_dir, data_dir="ml_data"):
    """
    Temporal distributions by predicted class.
    Uses sunrise/sunset-based period proportions if 'ml_data/period_features.csv' exists,
    otherwise falls back to fixed clock windows. Safe against missing hours.
    """
    _set_style(); _ensure_dir(output_dir)

    pred_col = 'predicted_pattern' if 'predicted_pattern' in results_df.columns else 'prediction'
    hd = hourly_data.merge(results_df[['species', pred_col]], on='species', how='left')

    solar = _load_solar_periods_table(data_dir)
    use_solar = solar is not None

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, cls in enumerate(CLASSES):
        ax = axes[i]

        # species predicted as this class
        sp = results_df.loc[results_df[pred_col] == cls, 'species']
        if len(sp) == 0:
            ax.text(0.5, 0.5, f'No species for {PRETTY[cls]}', ha='center', va='center')
            ax.axis('off'); continue

        if use_solar:
            sub = solar[solar['species'].isin(sp)]
            if len(sub) == 0:
                ax.text(0.5, 0.5, f'No solar data ({PRETTY[cls]})', ha='center', va='center')
                ax.axis('off'); continue

            means = sub[['night_activity','dawn_activity','day_activity','dusk_activity']].mean()
            labels = ['Night','Dawn','Day','Dusk']
            vals = means[['night_activity','dawn_activity','day_activity','dusk_activity']].values

            ax.bar(labels, vals, edgecolor='black', linewidth=1.5, color=[COLORS[cls]]*4, alpha=0.85)
            ax.set_ylim(0, 1)
            for x, v in zip(labels, vals):
                ax.text(x, v + 0.02, f'{v:.0%}', ha='center', va='bottom', fontsize=10, weight='bold')
            ax.set_title(f'{PRETTY[cls]} — Solar Periods')
            ax.set_ylabel('Proportion of activity'); ax.grid(axis='y', alpha=0.2)

        else:
            sub = hd[hd['species'].isin(sp)]
            if len(sub) == 0:
                ax.text(0.5, 0.5, f'No species for {PRETTY[cls]}', ha='center', va='center')
                ax.axis('off'); continue

            mean_curve = sub.groupby('hour')['activity_density'].mean()
            mean_curve = _reindex_24(mean_curve)

            night_hours = list(range(0, 6)) + list(range(18, 24))
            day_hours   = list(range(6, 18))
            dawn_hours  = [4, 5, 6, 7]
            dusk_hours  = [17, 18, 19, 20]

            night = _sum_hours(mean_curve, night_hours)
            dawn  = _sum_hours(mean_curve, dawn_hours)
            day   = _sum_hours(mean_curve, day_hours)
            dusk  = _sum_hours(mean_curve, dusk_hours)
            total = night + dawn + day + dusk
            vals = np.array([night, dawn, day, dusk]) / total if total > 0 else np.zeros(4)
            labels = ['Night','Dawn','Day','Dusk']

            ax.bar(labels, vals, edgecolor='black', linewidth=1.5, color=[COLORS[cls]]*4, alpha=0.85)
            ax.set_ylim(0, 1)
            for x, v in zip(labels, vals):
                ax.text(x, v + 0.02, f'{v:.0%}', ha='center', va='bottom', fontsize=10, weight='bold')
            ax.set_title(f'{PRETTY[cls]} — Clock Windows')
            ax.set_ylabel('Proportion of activity'); ax.grid(axis='y', alpha=0.2)

    plt.suptitle('Temporal Distributions by Predicted Class', y=0.98, weight='bold')
    out = f"{output_dir}/temporal_distributions.png"
    plt.tight_layout(); plt.savefig(out, facecolor='white'); plt.close()
    print(f"temporal distributions saved to {out}")



# =========================================
# 7) Model comparison
# =========================================
def plot_model_comparison(model_results, output_dir):
    _set_style(); _ensure_dir(output_dir)
    rows = []
    for k, v in model_results.items():
        acc = v.get('accuracy', None)
        if acc is not None:
            rows.append((k, acc))
    if not rows:
        print("model comparison skipped (no accuracies)"); return
    df = pd.DataFrame(rows, columns=['model', 'accuracy'])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df, x='model', y='accuracy', ax=ax)
    ax.set_ylim(0, 1); ax.set_ylabel("Accuracy")
    for p in ax.patches:
        ax.text(p.get_x()+p.get_width()/2, p.get_height()+0.02, f"{p.get_height():.1%}", ha='center', va='bottom', fontsize=11, weight='bold')
    ax.set_title("Model Performance Comparison")
    out = f"{output_dir}/model_comparison.png"
    plt.tight_layout(); plt.savefig(out, facecolor='white'); plt.close()
    print(f"model comparison saved to {out}")



# =========================================
# 8) Misclassification analysis
# =========================================
def plot_misclassification_analysis(results_df, hourly_data, output_dir):
    _set_style(); _ensure_dir(output_dir)
    if not {'pattern','predicted_pattern'}.issubset(results_df.columns):
        print("misclassification analysis skipped (needs 'pattern' and 'predicted_pattern')")
        return
    df = results_df.dropna(subset=['pattern','predicted_pattern']).copy()
    df['correct'] = (df['pattern'].str.lower() == df['predicted_pattern'].str.lower())

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(data=df, x='predicted_pattern', hue='correct', palette=['#e74c3c','#2ecc71'], ax=ax,
                  order=CLASSES)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Count")
    ax.set_xticklabels([PRETTY[c] for c in CLASSES])
    ax.set_title("Misclassification Breakdown (by predicted class)")
    out = f"{output_dir}/misclassification_breakdown.png"
    plt.tight_layout(); plt.savefig(out, facecolor='white'); plt.close()
    print(f"misclassification analysis saved to {out}")



# =========================================
# 9) Peak hour analysis
# =========================================
def plot_peak_hour_analysis(hourly_data, predictions, output_dir):
    """Analyze peak activity hours (DIURNAL + NOCTURNAL only) with original aesthetics."""
    _set_style()  # use the module's talk/whitegrid look

    # --- choose predictions column
    if 'predicted_pattern' in predictions.columns:
        pred_col = 'predicted_pattern'
    elif 'ensemble_prediction' in predictions.columns:
        pred_col = 'ensemble_prediction'
    else:
        pred_col = 'prediction'

    # --- filter to diurnal/nocturnal (case-insensitive)
    preds = predictions.copy()
    preds[pred_col] = preds[pred_col].astype(str).str.lower()
    preds = preds[preds[pred_col].isin(['nocturnal', 'diurnal'])].copy()
    if preds.empty:
        print("peak hour analysis skipped (no nocturnal/diurnal rows)")
        return

    # --- canonicalize species keys (helps the join)
    def _canon(s):
        return (s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip())
    preds['species']  = _canon(preds['species'])
    hourly = hourly_data.copy()
    hourly['species'] = _canon(hourly['species'])

    # --- compute per-species peak hour/value
    peak_rows = []
    for sp in preds['species'].unique():
        sd = hourly[hourly['species'] == sp]
        if sd.empty:
            continue
        curve = (sd.groupby('hour')['activity_density']
                   .mean()
                   .reindex(range(24), fill_value=0))
        if curve.max() <= 0:
            continue
        peak_rows.append({
            'species': sp,
            'peak_hour': int(curve.idxmax()),
            'peak_value': float(curve.max()),
            'pattern': preds.loc[preds['species'] == sp, pred_col].iloc[0]
        })

    peak_df = pd.DataFrame(peak_rows)
    if peak_df.empty:
        print("peak hour analysis skipped (no overlapping species with hourly data)")
        return

    # --- figure & axes (match original 2x2 layout, with a proper polar TL panel)
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2)
    ax_polar   = fig.add_subplot(gs[0, 0], projection='polar')
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_density = fig.add_subplot(gs[1, 0])
    ax_box     = fig.add_subplot(gs[1, 1])

    # consistent colors like the original module
    color_map = {'nocturnal': COLORS.get('nocturnal', '#2c3e50'),
                 'diurnal'  : COLORS.get('diurnal',   '#f39c12')}

    # ---- (1) circular histogram of peak hours
    theta = np.linspace(0, 2*np.pi, 24, endpoint=False)
    for pat in ['nocturnal', 'diurnal']:
        hours = peak_df.loc[peak_df['pattern'] == pat, 'peak_hour'].to_numpy()
        if hours.size == 0:
            continue
        hist, _ = np.histogram(hours, bins=24, range=(0, 24))
        hist = hist / hist.max() if hist.max() > 0 else hist
        ax_polar.bar(theta, hist, width=2*np.pi/24, alpha=0.5,
                     label=PRETTY.get(pat, pat.title()),
                     color=color_map[pat], edgecolor='none')

    # orient like original: 0 at top, clockwise
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)
    ax_polar.set_xticks(theta)
    ax_polar.set_xticklabels(range(24))
    ax_polar.set_title('peak hour distribution (circular)', y=1.08, fontweight='bold')
    ax_polar.legend(loc='upper right', bbox_to_anchor=(1.30, 1.10), frameon=False)

    # ---- (2) peak hour vs peak value (scatter)
    for pat in ['nocturnal', 'diurnal']:
        sub = peak_df[peak_df['pattern'] == pat]
        if not sub.empty:
            ax_scatter.scatter(sub['peak_hour'], sub['peak_value'],
                               s=30, alpha=0.6, linewidth=0.5,
                               edgecolors='black', zorder=3,
                               label=PRETTY.get(pat, pat.title()),
                               color=color_map[pat])
    ax_scatter.set_xlabel('peak hour')
    ax_scatter.set_ylabel('peak activity value')
    ax_scatter.set_xticks(range(0, 24, 3))
    ax_scatter.set_title('peak hour vs intensity', fontweight='bold')
    ax_scatter.legend(frameon=False)
    ax_scatter.grid(alpha=0.25)

    # ---- (3) peak hour density (KDE if SciPy available, else histogram)
    try:
        from scipy import stats
        have_scipy = True
    except Exception:
        have_scipy = False

    xs = np.linspace(0, 23, 200)
    for pat in ['nocturnal', 'diurnal']:
        vals = peak_df.loc[peak_df['pattern'] == pat, 'peak_hour'].to_numpy()
        if vals.size == 0:
            continue
        if have_scipy and vals.size >= 2:
            kde = stats.gaussian_kde(vals)
            ax_density.plot(xs, kde(xs), linewidth=2.5, label=PRETTY.get(pat, pat.title()),
                            color=color_map[pat])
        else:
            ax_density.hist(vals, bins=np.arange(25)-0.5, density=True, alpha=0.35,
                            label=PRETTY.get(pat, pat.title()),
                            color=color_map[pat], edgecolor='black', linewidth=0.3)
    ax_density.set_xlabel('hour of day')
    ax_density.set_ylabel('density')
    ax_density.set_xlim(0, 23)
    ax_density.set_title('peak hour density by pattern', fontweight='bold')
    ax_density.legend(frameon=False)
    ax_density.grid(alpha=0.25)

    # ---- (4) peak width @ 50% max (boxplot)
    peak_widths, patterns = [], []
    for sp in preds['species'].unique()[:100]:  # modest cap for speed
        sd = hourly[hourly['species'] == sp]
        if sd.empty:
            continue
        curve = (sd.groupby('hour')['activity_density']
                   .mean()
                   .reindex(range(24), fill_value=0))
        if curve.max() <= 0:
            continue
        width = int((curve > (curve.max()/2.0)).sum())
        peak_widths.append(width)
        patterns.append(preds.loc[preds['species'] == sp, pred_col].iloc[0])

    width_df = pd.DataFrame({'width': peak_widths, 'pattern': patterns})
    if not width_df.empty:
        order = [c for c in ['nocturnal', 'diurnal'] if c in width_df['pattern'].unique()]
        sns.boxplot(data=width_df, x='pattern', y='width', order=order, ax=ax_box,
                    palette=[color_map[c] for c in order])
        ax_box.set_xticklabels([PRETTY.get(c, c.title()) for c in order])
    ax_box.set_ylabel('peak width (hours > 50% max)')
    ax_box.set_title('activity peak width by pattern', fontweight='bold')
    ax_box.grid(axis='y', alpha=0.25)

    plt.suptitle('Peak Activity Hour Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/peak_hour_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# =========================================
# 10) Bimodality analysis (simple)
# =========================================
def _count_local_peaks(arr):
    peaks = 0
    for i in range(1, len(arr)-1):
        if arr[i] > arr[i-1] and arr[i] > arr[i+1] and arr[i] > 0.3*arr.max():
            peaks += 1
    return peaks

def plot_bimodality_analysis(hourly_data, results_df, output_dir):
    _set_style(); _ensure_dir(output_dir)
    pred_col = 'predicted_pattern' if 'predicted_pattern' in results_df.columns else 'prediction'
    merged = hourly_data.merge(results_df[['species', pred_col]], on='species', how='left')

    rows = []
    for sp, g in merged.groupby('species'):
        curve = _reindex_24(g.groupby('hour')['activity_density'].mean())
        peaks = _count_local_peaks(curve.values)
        rows.append((sp, peaks))
    bi = pd.DataFrame(rows, columns=['species','n_peaks']).merge(results_df[['species', pred_col]], on='species', how='left')
    bi['bimodal'] = (bi['n_peaks'] >= 2)

    agg = bi.groupby(pred_col)['bimodal'].mean().reindex(CLASSES).fillna(0)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.barplot(x=[PRETTY[c] for c in CLASSES], y=agg.values, ax=ax)
    ax.set_ylim(0,1); ax.set_ylabel('% Bimodal')
    for i, v in enumerate(agg.values):
        ax.text(i, v+0.02, f"{v:.0%}", ha='center', va='bottom', fontsize=11, weight='bold')
    ax.set_title("Bimodal Activity (≥2 peaks) by Predicted Class")
    out = f"{output_dir}/bimodality_by_class.png"
    plt.tight_layout(); plt.savefig(out, facecolor='white'); plt.close()
    print(f"bimodality analysis saved to {out}")



# =========================================
# 11) Learning curves (best effort)
# =========================================
def plot_learning_curves(model_results, X_train, y_train, output_dir):
    from sklearn.model_selection import learning_curve
    _set_style(); _ensure_dir(output_dir)

    if not model_results:
        print("learning curves skipped (no models)"); return
    # pick best by accuracy
    best_key = max(model_results, key=lambda k: model_results[k].get('accuracy', -1))
    model = model_results[best_key].get('model', None)
    if model is None:
        print("learning curves skipped (no model object)"); return

    try:
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=5, train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1
        )
    except Exception as e:
        print(f"learning curves skipped ({e})"); return

    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(train_sizes, train_scores.mean(axis=1), marker='o', label='Train')
    ax.plot(train_sizes, val_scores.mean(axis=1), marker='o', label='CV')
    ax.fill_between(train_sizes, train_scores.mean(axis=1)-train_scores.std(axis=1),
                    train_scores.mean(axis=1)+train_scores.std(axis=1), alpha=0.2)
    ax.fill_between(train_sizes, val_scores.mean(axis=1)-val_scores.std(axis=1),
                    val_scores.mean(axis=1)+val_scores.std(axis=1), alpha=0.2)
    ax.set_xlabel("Training examples"); ax.set_ylabel("Accuracy"); ax.set_ylim(0,1)
    ax.set_title(f"Learning Curves — {best_key}"); ax.legend()

    out = f"{output_dir}/learning_curves_{best_key}.png"
    plt.tight_layout(); plt.savefig(out, facecolor='white'); plt.close()
    print(f"learning curves saved to {out}")



# =========================================
# 12) ROC curves (best effort, one-vs-rest)
# =========================================
def plot_roc_curves(model_results, X_test, y_test, output_dir):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc
    _set_style(); _ensure_dir(output_dir)

    # pick a model with predict_proba
    cand = [(k, v['model']) for k, v in model_results.items() if hasattr(v.get('model', None), "predict_proba")]
    if not cand:
        print("roc curves skipped (no model with predict_proba)"); return
    name, model = cand[0]

    try:
        y_prob = model.predict_proba(X_test)
        classes = getattr(model, "classes_", None)
        if classes is None: raise ValueError("model lacks classes_")
        y_true_bin = label_binarize(y_test, classes=classes)
    except Exception as e:
        print(f"roc curves skipped ({e})"); return

    fig, ax = plt.subplots(figsize=(8,6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        ax.plot(fpr, tpr, lw=2, label=str(cls) + f" (AUC={auc(fpr,tpr):.2f})")
    ax.plot([0,1], [0,1], 'k--', lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {name} (OvR)")
    ax.legend()

    out = f"{output_dir}/roc_curves_{name}.png"
    plt.tight_layout(); plt.savefig(out, facecolor='white'); plt.close()
    print(f"roc curves saved to {out}")



# =========================================
# 13) Activity heatmap by class
# =========================================
def plot_activity_heatmap(hourly_data, results_df, output_dir):
    _set_style(); _ensure_dir(output_dir)
    pred_col = "predicted_pattern" if "predicted_pattern" in results_df.columns else "prediction"

    # only keep classes that are actually present after filtering
    desired = ["nocturnal", "diurnal", "crepuscular", "cathemeral"]
    present_mask = results_df[pred_col].astype(str).str.lower().isin(desired)
    classes = [c for c in desired if c in results_df.loc[present_mask, pred_col].str.lower().unique()]
    if not classes:
        print("activity heatmap skipped (no classes present after filtering)")
        return

    # inner join so we only keep overlapping species
    merged = hourly_data.merge(results_df[["species", pred_col]], on="species", how="inner")
    if merged.empty:
        print("activity heatmap skipped (no species overlap between results and hourly_data)")
        return

    # mean activity per class per hour
    g = merged.groupby([pred_col, "hour"])["activity_density"].mean()
    if g.empty:
        print("activity heatmap skipped (no activity rows after groupby)")
        return

    table = (g.unstack("hour")
               .reindex(index=classes)                           # only present classes
               .pipe(lambda df: df.reindex(columns=sorted(df.columns)))  # hours ascending
               .fillna(0))
    if table.empty or table.shape[1] == 0:
        print("activity heatmap skipped (no hour columns present)")
        return

    # normalize rows to max=1 for visibility
    table = table.div(table.max(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(table, cmap="mako", cbar_kws={"label": "Normalized activity"}, ax=ax)
    ax.set_yticklabels([PRETTY.get(c, c.title()) for c in table.index])
    ax.set_xlabel("Hour of day"); ax.set_title("Activity Heatmap by Predicted Class")

    out = f"{output_dir}/activity_heatmap.png"
    plt.tight_layout(); plt.savefig(out, facecolor="white"); plt.close()
    print(f"activity heatmap saved to {out}")

# =========================================
# 14) Text summary report
# =========================================
def create_summary_report(results_df, model_results, output_dir):
    _ensure_dir(output_dir)
    lines = []
    lines.append("SUMMARY REPORT\n")
    lines.append("==============================\n\n")
    # model accuracies
    if model_results:
        lines.append("Model accuracies:\n")
        for k, v in model_results.items():
            acc = v.get('accuracy', None)
            if acc is not None:
                lines.append(f"  - {k}: {acc:.3f}\n")
        lines.append("\n")
    # class distribution
    pred_col = 'predicted_pattern' if 'predicted_pattern' in results_df.columns else 'prediction'
    if pred_col in results_df:
        dist = results_df[pred_col].value_counts(normalize=True).reindex(CLASSES).fillna(0)
        lines.append("Predicted class distribution:\n")
        for cls, p in dist.items():
            lines.append(f"  - {cls}: {p:.1%}\n")
        lines.append("\n")
    # confidence stats
    if 'confidence' in results_df.columns:
        c = results_df['confidence']
        lines.append("Confidence statistics:\n")
        lines.append(f"  - mean: {c.mean():.3f}\n")
        lines.append(f"  - >0.7: {(c>0.7).sum()} ({(c>0.7).mean():.1%})\n")
        lines.append(f"  - >0.9: {(c>0.9).sum()} ({(c>0.9).mean():.1%})\n")
        lines.append("\n")

    out = Path(output_dir) / "summary_report.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"summary report written to {out}")

if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        description="Make poster-ready figures from ML results + hourly activity data."
    )
    parser.add_argument("--results", required=True,
                        help="CSV of ML results per species (must include 'species' and 'predicted_pattern' or 'prediction').")
    parser.add_argument("--hourly", required=True,
                        help="CSV with per-species hourly activity (columns: species,hour,activity_density).")
    parser.add_argument("--out", default="poster_figures",
                        help="Output folder for images (default: poster_figures).")
    parser.add_argument("--data-dir", default="ml_data",
                        help="Folder that may contain period_features.csv for solar-aware plots (default: ml_data).")
    parser.add_argument("--n-examples", type=int, default=20,
                        help="How many species panels to sample for the species-curves figure (default: 20).")
    parser.add_argument("--pred-col", default=None,
                        help="If your predictions column is custom, provide its name (will be renamed to 'predicted_pattern').")
    parser.add_argument("--true-col", default=None,
                        help="If you have ground truth labels, provide its name (will be renamed to 'pattern').")
    parser.add_argument("--confidence-col", default=None,
                        help="If you have a custom confidence column name, provide it (renamed to 'confidence').")
    parser.add_argument("--two-classes", action="store_true",
                        help="Keep only nocturnal + diurnal everywhere (drop crepuscular & cathemeral).")

    args = parser.parse_args()

    # ---- Load data
    try:
        results_df = pd.read_csv(args.results)
    except Exception as e:
        print(f"Failed to read --results CSV: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        hourly_data = pd.read_csv(args.hourly)
    except Exception as e:
        print(f"Failed to read --hourly CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # ---- Normalize column names in results_df
    if args.pred_col and args.pred_col in results_df.columns:
        results_df = results_df.rename(columns={args.pred_col: "predicted_pattern"})
    elif "predicted_pattern" not in results_df.columns and "prediction" in results_df.columns:
        results_df = results_df.rename(columns={"prediction": "predicted_pattern"})

    if args.true_col and args.true_col in results_df.columns:
        results_df = results_df.rename(columns={args.true_col: "pattern"})

    if args.confidence_col and args.confidence_col in results_df.columns:
        results_df = results_df.rename(columns={args.confidence_col: "confidence"})

    # required columns check
    need_cols = {"species", "predicted_pattern"}
    if not need_cols.issubset(results_df.columns):
        print(f"Results CSV must include columns: {need_cols}", file=sys.stderr)
        sys.exit(1)

    # lower-case labels for consistency
    results_df["predicted_pattern"] = results_df["predicted_pattern"].astype(str).str.lower()
    if "pattern" in results_df.columns:
        results_df["pattern"] = results_df["pattern"].astype(str).str.lower()

    # ---- Optional: restrict to two classes
    if args.two_classes:
        keep = {"nocturnal", "diurnal"}
        results_df = results_df[results_df["predicted_pattern"].isin(keep)].copy()
        if "pattern" in results_df.columns:
            results_df = results_df[results_df["pattern"].isin(keep)].copy()

    # ---- Normalize hourly_data columns
    # allow 'activity' as alias for 'activity_density'
    if "activity_density" not in hourly_data.columns and "activity" in hourly_data.columns:
        hourly_data = hourly_data.rename(columns={"activity": "activity_density"})
    # ensure needed columns
    need_hourly = {"species", "hour", "activity_density"}
    if not need_hourly.issubset(hourly_data.columns):
        print(f"Hourly CSV must include columns: {need_hourly}", file=sys.stderr)
        sys.exit(1)

    # coerce hour to integer [0..23]
    hourly_data["hour"] = pd.to_numeric(hourly_data["hour"], errors="coerce").fillna(0).astype(int).clip(0, 23)

    # ---- Make output dir
    _ensure_dir(args.out)

    # ---- Call plots that work with results + hourly only
    print("creating plots...")
    plot_activity_patterns_by_class(hourly_data, results_df, args.out)
    plot_temporal_distributions(hourly_data, results_df, args.out, data_dir=args.data_dir)
    plot_species_curves(hourly_data, results_df, args.out, n_examples=args.n_examples)
    plot_peak_hour_analysis(hourly_data, results_df, args.out)
    plot_bimodality_analysis(hourly_data, results_df, args.out)
    plot_activity_heatmap(hourly_data, results_df, args.out)
    create_summary_report(results_df, model_results={}, output_dir=args.out)

    # Optional: only run if you have matching columns/data
    if {"pattern", "predicted_pattern"}.issubset(results_df.columns):
        plot_detailed_confusion_matrix(results_df, args.out)
    if "confidence" in results_df.columns:
        plot_confidence_analysis(results_df, args.out)

    print("done.")
