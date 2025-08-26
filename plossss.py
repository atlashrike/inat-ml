#!/usr/bin/env python3
"""
some of these outputs are REALLY UGLY but I lost the code to some of my nice plots
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ======================
# Aesthetics & palettes
# ======================
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelsize": 13,
    "axes.titlesize": 15,
    "figure.titlesize": 17,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
    "font.family": "sans-serif",
})

# Okabe-Ito (colorblind-friendly)
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "verm": "#D55E00",
    "purple": "#CC79A7",
}
COLORS = {
    "nocturnal": OKABE_ITO["blue"],
    "diurnal": OKABE_ITO["orange"],
    "crepuscular": OKABE_ITO["purple"],
    "cathemeral": OKABE_ITO["green"],
}

CLASSES_ALL = ["nocturnal", "diurnal", "crepuscular", "cathemeral"]
PRETTY = {
    "nocturnal": "Nocturnal",
    "diurnal": "Diurnal",
    "crepuscular": "Crepuscular",
    "cathemeral": "Cathemeral",
}

def _ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def _clean_species(s):
    return s.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

def _pick_pred_col(df):
    for c in ["predicted_pattern", "ensemble_prediction", "prediction"]:
        if c in df.columns:
            return c
    raise ValueError("Results CSV needs one of: predicted_pattern / ensemble_prediction / prediction")

def _reindex_24(series: pd.Series) -> pd.Series:
    return series.reindex(range(24), fill_value=0)

def _sum_hours(series: pd.Series, hours) -> float:
    return _reindex_24(series).reindex(hours, fill_value=0).sum()

def _load_period_table(data_dir: str):
    """Optional solar-aware per-species period proportions (R export)."""
    p = Path(data_dir) / "period_features.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    need = {"species", "night_activity", "day_activity", "dawn_activity", "dusk_activity"}
    if not need.issubset(df.columns):
        return None
    cols = ["night_activity", "day_activity", "dawn_activity", "dusk_activity"]
    s = df[cols].sum(axis=1).replace(0, np.nan)
    for c in cols:
        df[c] = df[c] / s
    return df.fillna(0)

def _period_windows():
    """Fallback fixed windows if no solar table present."""
    return {
        "night": list(range(0, 6)) + list(range(18, 24)),
        "day": list(range(6, 18)),
        "dawn": [4, 5, 6, 7],
        "dusk": [17, 18, 19, 20],
    }

def _nice_int(x):
    try:
        return int(x)
    except Exception:
        return np.nan


def load_data(results_csv, hourly_csv, data_dir, two_classes=False):
    results = pd.read_csv(results_csv)
    hourly = pd.read_csv(hourly_csv)

    # normalize columns
    pred_col = _pick_pred_col(results)

    if "activity_density" not in hourly.columns and "activity" in hourly.columns:
        hourly = hourly.rename(columns={"activity": "activity_density"})

    need_hourly = {"species", "hour", "activity_density"}
    if not need_hourly.issubset(hourly.columns):
        raise ValueError(f"Hourly CSV must have columns: {need_hourly}")

    results["species"] = _clean_species(results["species"])
    results[pred_col] = results[pred_col].astype(str).str.lower()

    # Preserve NaN for pattern; only lowercase the strings
    if "pattern" in results.columns:
        results["pattern"] = pd.Series(results["pattern"], dtype="string").str.lower()

    if "confidence" in results.columns:
        results["confidence"] = pd.to_numeric(results["confidence"], errors="coerce")

    if two_classes:
        keep = {"nocturnal", "diurnal"}
        results = results[results[pred_col].isin(keep)].copy()
        if "pattern" in results.columns:
            # keep rows where pattern is diurnal/nocturnal OR genuinely missing
            pat = pd.Series(results["pattern"], dtype="string")
            results = results[pat.isna() | pat.isin(keep)].copy()

    hourly["species"] = _clean_species(hourly["species"])
    hourly["hour"] = pd.to_numeric(hourly["hour"], errors="coerce").apply(_nice_int).clip(0, 23)
    hourly["activity_density"] = pd.to_numeric(hourly["activity_density"], errors="coerce").fillna(0)

    solar = _load_period_table(data_dir)
    return results, hourly, pred_col, solar


def fig_background(results, hourly, pred_col, outdir):
    """Uses real activity curves averaged across species to show hour bias."""
    _ensure_dir(outdir)

    # mean across species (each species contributes equally)
    m = (
        hourly.groupby(["species", "hour"])["activity_density"]
        .mean()
        .groupby("hour")
        .mean()
    )
    m = _reindex_24(m)
    hours = np.arange(24)

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.35)

    # Panel A: average hourly activity
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.bar(hours, m.values * 100, color=OKABE_ITO["sky"], edgecolor="black", linewidth=0.5)
    ax1.axvspan(-0.5, 5.5, alpha=0.2, color="gray", label="Night hours")
    ax1.axvspan(18.5, 23.5, alpha=0.2, color="gray")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylabel("% of Mean Activity (species-mean)")
    ax1.set_title("Average Hourly Activity Across Species (species-weighted)")
    day = list(range(6, 18))
    night = list(range(0, 6)) + list(range(18, 24))
    pct_day = 100 * _sum_hours(m, day) / (_sum_hours(m, day) + _sum_hours(m, night) + 1e-12)
    ax1.text(
        12,
        ax1.get_ylim()[1] * 0.85,
        f"{pct_day:.1f}% of activity in daylight hours",
        ha="center",
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor=OKABE_ITO["yellow"], alpha=0.5),
    )

    # Panel B: placeholders for organisms (swap for images if you like)
    from matplotlib.patches import Circle
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title("Study Organisms")
    ax2.axis("off")
    species_boxes = [
        ("Moths (Nocturnal)", 0.5, 0.85, COLORS["nocturnal"]),
        ("Butterflies (Diurnal)", 0.5, 0.65, COLORS["diurnal"]),
        ("Fireflies (Crepuscular)", 0.5, 0.45, COLORS["crepuscular"]),
        ("Beetles (Variable)", 0.5, 0.25, COLORS["cathemeral"]),
    ]
    for name, x, y, color in species_boxes:
        circle = Circle((x - 0.2, y), 0.08, transform=ax2.transAxes, facecolor=color, edgecolor="black", linewidth=1.5, alpha=0.7)
        ax2.add_patch(circle)
        ax2.text(x + 0.05, y, name, transform=ax2.transAxes, fontsize=11, va="center")

    # Panel C: processing pipeline (textual)
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis("off")
    ax3.set_title("Data Processing Pipeline")
    steps = [
        "Raw iNat observations per species/hour",
        "Solar-aware period assignment (if available)",
        "Normalize to activity density within day",
        "Aggregate to per-species 24 h curve",
    ]
    x0 = 0.1
    for i, s in enumerate(steps):
        ax3.text(
            x0 + i * 0.22,
            0.5,
            f"{i+1}. {s}",
            transform=ax3.transAxes,
            bbox=dict(
                boxstyle="round",
                facecolor="#E3F2FD" if i % 2 == 0 else "#C8E6C9",
                edgecolor="black",
            ),
        )
        if i < len(steps) - 1:
            ax3.annotate(
                "",
                xy=(x0 + (i + 1) * 0.22 - 0.02, 0.5),
                xytext=(x0 + i * 0.22 + 0.08, 0.5),
                arrowprops=dict(arrowstyle="->", lw=2, color="black"),
                transform=ax3.transAxes,
            )

    # Panel D: distribution of per-species totals (proxy for sample size)
    ax4 = fig.add_subplot(gs[2, :2])
    if "count" in hourly.columns:
        sizes = hourly.groupby("species")["count"].sum()
    else:
        sizes = hourly.groupby("species")["activity_density"].sum()
    sizes = sizes.replace([np.inf, -np.inf], np.nan).dropna()
    ax4.hist(sizes, bins=40, color=OKABE_ITO["verm"], edgecolor="black", alpha=0.7)
    ax4.set_title("Per-species sampling (proxy)")
    ax4.set_xlabel("Total signal")
    ax4.set_ylabel("Species count")

    # Panel E: class distribution from real results
    ax5 = fig.add_subplot(gs[2, 2])
    counts = (
        results[pred_col]
        .value_counts()
        .reindex(["nocturnal", "diurnal", "crepuscular", "cathemeral"])
        .fillna(0)
        .astype(int)
    )
    ax5.bar(
        [PRETTY.get(c, c.title()) for c in counts.index],
        counts.values,
        color=[COLORS.get(c, "#888") for c in counts.index],
        edgecolor="black",
    )
    ax5.set_title("Predicted class distribution")
    ax5.set_ylabel("Species")

    plt.suptitle("Background & Context (using real data)", y=0.98, weight="bold")
    plt.tight_layout()
    out = Path(outdir) / "background_real.png"
    plt.savefig(out)
    plt.close()
    print(f"  - saved {out}")


def fig_dawn_dusk(results, hourly, pred_col, solar, outdir):
    _ensure_dir(outdir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # A) visual definition
    ax = axes[0]
    hours = np.arange(24)
    for h in hours:
        if 5 <= h < 7:      # Dawn
            color, alpha = "#FFB74D", 0.35
        elif 7 <= h < 18:   # Day
            color, alpha = "#FFF59D", 0.25
        elif 18 <= h < 20:  # Dusk
            color, alpha = "#FF8A65", 0.35
        else:               # Night
            color, alpha = "#424242", 0.28
        ax.bar(h, 1, width=1, color=color, alpha=alpha)
    ax.axvline(5, color="red", linestyle="--", linewidth=2, label="Twilight start")
    ax.axvline(7, color="orange", linestyle="--", linewidth=2, label="Sunrise")
    ax.axvline(18, color="orange", linestyle="--", linewidth=2, label="Sunset")
    ax.axvline(20, color="red", linestyle="--", linewidth=2, label="Twilight end")
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Hour of Day")
    ax.set_title("Dawn/Dusk Definition (~Â±1 h from sunrise/sunset)")
    ax.legend(fontsize=10)
    ax.set_xticks(range(0, 24, 3))

    # B) cohort averages from real data: proportions by period for diurnal vs nocturnal
    ax = axes[1]
    vals = {}
    if solar is not None:
        tmp = results.merge(solar, on="species", how="left")
        tmp = tmp.dropna(subset=["night_activity", "day_activity", "dawn_activity", "dusk_activity"])
        if not tmp.empty:
            for cls in ["nocturnal", "diurnal"]:
                sub = tmp[tmp[pred_col] == cls][["night_activity", "dawn_activity", "day_activity", "dusk_activity"]]
                if not sub.empty:
                    v = sub.mean()
                    v = v.rename({
                        "night_activity": "night",
                        "dawn_activity": "dawn",
                        "day_activity": "day",
                        "dusk_activity": "dusk",
                    })
                    vals[cls] = v
    if not vals:  # fallback to fixed windows
        windows = _period_windows()
        merged = hourly.merge(results[["species", pred_col]], on="species", how="left")
        for cls in ["nocturnal", "diurnal"]:
            sp = merged[merged[pred_col] == cls]
            if sp.empty:
                continue
            curve = sp.groupby("hour")["activity_density"].mean()
            curve = _reindex_24(curve)
            night = _sum_hours(curve, windows["night"])
            dawn = _sum_hours(curve, windows["dawn"])
            day = _sum_hours(curve, windows["day"])
            dusk = _sum_hours(curve, windows["dusk"])
            tot = night + dawn + day + dusk
            if tot > 0:
                vals[cls] = pd.Series(
                    [night, dawn, day, dusk],
                    index=["night", "dawn", "day", "dusk"],
                ) / tot

    labels = ["Night", "Dawn", "Day", "Dusk"]
    x = np.arange(len(labels))
    w = 0.35
    plotted = False
    order = ["night", "dawn", "day", "dusk"]
    for i, cls in enumerate(["nocturnal", "diurnal"]):
        v = vals.get(cls, None)
        if v is None:
            continue
        plotted = True
        ax.bar(
            x + (i - 0.5) * w,
            v[order].values,
            width=w,
            color=COLORS[cls],
            alpha=0.8,
            edgecolor="black",
            label=PRETTY[cls],
        )
        for xi, vv in zip(x + (i - 0.5) * w, v[order].values):
            ax.text(xi, vv + 0.02, f"{vv:.0%}", ha="center", va="bottom", fontsize=10, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Proportion of activity")
    ax.set_title("Temporal Period Proportions (cohort means)")
    if plotted:
        ax.legend()

    plt.suptitle("Temporal Period Definitions & Cohort Averages", weight="bold")
    out = Path(outdir) / "dawn_dusk_real.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"  - saved {out}")


def fig_novel_discoveries(results, hourly, pred_col, outdir, n=6):
    _ensure_dir(outdir)
    if "pattern" not in results.columns:
        print("  - novel discoveries: skipped (no 'pattern' column)")
        return

    # Treat these as missing-labeled tokens
    missing_tokens = {"", "unknown", "na", "n/a", "none"}
    pat = pd.Series(results["pattern"], dtype="string")  # preserves <NA>
    missing_mask = pat.isna() | pat.str.strip().isin(missing_tokens)

    keep_calls = results[pred_col].astype(str).str.lower().isin(["nocturnal", "diurnal"])
    cand = results[missing_mask & keep_calls].copy()

    if cand.empty:
        print("  - novel discoveries: no species without literature labels")
        return

    if "confidence" in cand.columns:
        cand = cand.sort_values("confidence", ascending=False, na_position="last")

    sel = cand.head(n)

    rows, cols = (2, int(np.ceil(n / 2)))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8), sharex=True, sharey=False)
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, r) in zip(axes, sel.iterrows()):
        sp, pat_ = r["species"], r[pred_col]
        curve = (
            hourly[hourly["species"] == sp]
            .groupby("hour")["activity_density"]
            .mean()
            .pipe(_reindex_24)
        )
        ax.bar(
            np.arange(24),
            curve.values,
            color=COLORS.get(pat_, "#888"),
            alpha=0.75,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.axvspan(-0.5, 5.5, alpha=0.12, color="gray")
        ax.axvspan(18.5, 23.5, alpha=0.12, color="gray")
        title = f"{sp}\nPred: {PRETTY.get(pat_, pat_.title())}"
        if "confidence" in sel.columns and pd.notna(r.get("confidence", np.nan)):
            title += f"  (conf {float(r['confidence']):.2f})"
        ax.set_title(title, fontsize=11)
        ax.set_xticks(range(0, 24, 6))
        ax.set_ylim(0, max(curve.values.max() * 1.15, 1e-3))
        ax.set_xlabel("Hour")
        ax.set_ylabel("Activity")

    for ax in axes[len(sel):]:
        ax.axis("off")

    plt.suptitle("Novel Discoveries (no literature label, real curves)", y=1.02, weight="bold")
    out = Path(outdir) / "novel_discoveries_real.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"  - saved {out}")


def fig_confusion_matrix(results, pred_col, outdir, drop_crepuscular=True):
    _ensure_dir(outdir)
    if "pattern" not in results.columns:
        print("  - confusion matrix: skipped (no 'pattern' column)")
        return

    df = results.dropna(subset=[pred_col]).copy()
    df[pred_col] = df[pred_col].astype(str).str.lower()

    # keep true labels that exist
    pat = pd.Series(results["pattern"], dtype="string")
    df["pattern"] = pat

    if drop_crepuscular:
        classes = ["nocturnal", "diurnal", "cathemeral"]
    else:
        classes = ["nocturnal", "diurnal", "cathemeral", "crepuscular"]

    df = df[df[pred_col].isin(classes)]
    df = df[df["pattern"].isin(classes)]  # drops <NA>

    if df.empty:
        print("  - confusion matrix: skipped (no overlapping classes after filtering)")
        return

    pivot = pd.crosstab(df["pattern"], df[pred_col], dropna=False).reindex(index=classes, columns=classes, fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        pivot,
        annot=True,
        fmt="d",
        cmap="RdYlGn_r",
        xticklabels=[PRETTY[c] for c in pivot.columns],
        yticklabels=[PRETTY[c] for c in pivot.index],
        cbar_kws={"label": "Count"},
        linewidths=1.5,
        linecolor="white",
        ax=ax,
        annot_kws={"fontsize": 12},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True (literature)")
    ax.set_title("Confusion Matrix (real labels)")
    acc = np.trace(pivot.values) / pivot.values.sum()
    ax.text(0.5, -0.12, f"Overall accuracy: {acc:.1%}", transform=ax.transAxes, ha="center", fontsize=12, fontweight="bold")

    out = Path(outdir) / "confusion_matrix_real.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"  - saved {out}")


def fig_activity_workflow(results, hourly, pred_col, outdir, example_species=None):
    _ensure_dir(outdir)

    # choose example species
    if example_species:
        sp = example_species
    else:
        merged = (
            hourly.groupby("species")["activity_density"]
            .sum()
            .reset_index()
            .rename(columns={"activity_density": "total"})
        )
        merged = merged.merge(results[["species", pred_col]], on="species", how="left")
        merged = merged[merged["total"] > 0].copy()
        if merged.empty:
            print("  - workflow: skipped (no nonzero species)")
            return
        order = pd.Categorical(
            merged[pred_col],
            categories=["diurnal", "nocturnal", "cathemeral", "crepuscular"],
            ordered=True,
        )
        merged = merged.assign(order=order).sort_values(["order", "total"], ascending=[True, False])
        sp = merged.iloc[0]["species"]

    # real curve
    raw_curve = (
        hourly[hourly["species"] == sp]
        .groupby("hour")["activity_density"]
        .mean()
        .pipe(_reindex_24)
    )

    peak_hour = int(raw_curve.idxmax())
    total = raw_curve.sum()
    windows = _period_windows()
    parts = {
        "Night": _sum_hours(raw_curve, windows["night"]) / total if total > 0 else 0,
        "Day": _sum_hours(raw_curve, windows["day"]) / total if total > 0 else 0,
        "Dawn": _sum_hours(raw_curve, windows["dawn"]) / total if total > 0 else 0,
        "Dusk": _sum_hours(raw_curve, windows["dusk"]) / total if total > 0 else 0,
    }
    half = raw_curve.max() / 2 if raw_curve.max() > 0 else 0
    width_hm = int((raw_curve > half).sum())

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4, hspace=0.35, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(range(24), raw_curve.values, color="gray", alpha=0.7, edgecolor="black")
    ax1.set_title("1. Species 24 h curve (real)")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Activity")
    ax1.set_xticks(range(0, 24, 6))

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(24), raw_curve.values, color=OKABE_ITO["yellow"], alpha=0.6, edgecolor="black")
    ax2.axvline(6, color="red", linestyle="--", label="Sunrise (proxy)")
    ax2.axvline(18, color="red", linestyle="--", label="Sunset (proxy)")
    ax2.set_title("2. Period landmarks")
    ax2.set_xlabel("Hour")
    ax2.set_xticks(range(0, 24, 6))
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    cols = []
    for h in range(24):
        if 6 <= h < 18:
            cols.append("#FFF59D")
        elif 5 <= h < 6 or 18 <= h < 20:
            cols.append("#FFB74D")
        else:
            cols.append("#424242")
    ax3.bar(range(24), raw_curve.values, color=cols, alpha=0.8, edgecolor="black")
    ax3.set_title("3. Period assignment")
    ax3.set_xlabel("Hour")
    ax3.set_xticks(range(0, 24, 6))

    ax4 = fig.add_subplot(gs[0, 3])
    dens = raw_curve / (raw_curve.sum() if raw_curve.sum() > 0 else 1)
    ax4.bar(range(24), dens.values, color=OKABE_ITO["green"], alpha=0.7, edgecolor="black")
    ax4.set_title("4. Normalized to density")
    ax4.set_xlabel("Hour")
    ax4.set_ylabel("Density")
    ax4.set_xticks(range(0, 24, 6))


    ax5 = fig.add_subplot(gs[1, :2])
    ax5.axis("off")
    ax5.set_title("5. Feature extraction (real numbers)", fontweight="bold", fontsize=14)
    lines = [
        f"- Species: {sp}",
        f"- Peak hour: {peak_hour:02d}:00",
        f"- Proportions: Day: {parts['Day']:.2f}, Night: {parts['Night']:.2f}, Dawn: {parts['Dawn']:.2f}, Dusk: {parts['Dusk']:.2f}",
        f"- Width at half-maximum: {width_hm} h",
    ]
    ax5.text(
        0.05,
        0.5,
        "\n".join(lines),
        transform=ax5.transAxes,
        fontsize=12,
        bbox=dict(boxstyle="round", facecolor="#E3F2FD", alpha=0.8),
    )

    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.axis("off")
    ax6.set_title("6. Classification result", fontweight="bold", fontsize=14)
    row = results.loc[results["species"] == sp].head(1)
    pred = row[pred_col].iloc[0] if not row.empty else "unknown"
    conf = None
    if "confidence" in row.columns and not row.empty:
        conf = row["confidence"].iloc[0]
        conf = float(conf) if pd.notna(conf) else None
    txt = f"{PRETTY.get(pred, pred.title())}"
    if conf is not None:
        txt += f"\nConfidence: {conf:.2f}"
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch(
        (0.1, 0.3),
        0.8,
        0.4,
        boxstyle="round,pad=0.05",
        transform=ax6.transAxes,
        facecolor="#C8E6C9",
        edgecolor="black",
        linewidth=2,
    )
    ax6.add_patch(box)
    ax6.text(0.5, 0.5, txt, transform=ax6.transAxes, ha="center", va="center", fontsize=16, fontweight="bold")

    plt.suptitle("Activity Curve Generation Workflow (real example)", weight="bold")
    out = Path(outdir) / "activity_curve_workflow_real.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"  - saved {out}")


def main():
    import argparse
    import sys

    p = argparse.ArgumentParser(description="Poster figures (real data).")
    p.add_argument("--results", required=True, help="CSV of predictions per species (must include 'species' and a prediction column).")
    p.add_argument("--hourly", required=True, help="CSV of per-species hourly activity ('species,hour,activity_density').")
    p.add_argument("--out", default="poster_figures", help="Output folder for images.")
    p.add_argument("--data-dir", default="ml_data", help="Folder that may contain period_features.csv (optional).")
    p.add_argument("--two-classes", action="store_true", help="Keep only diurnal + nocturnal.")
    p.add_argument("--example-species", default=None, help="Species name for the workflow panel (optional).")
    args = p.parse_args()

    try:
        results, hourly, pred_col, solar = load_data(
            args.results, args.hourly, args.data_dir, two_classes=args.two_classes
        )
    except Exception as e:
        print(f"Failed to load/normalize data: {e}", file=sys.stderr)
        sys.exit(1)

    _ensure_dir(args.out)
    print("Creating poster figures from REAL data...")
    fig_background(results, hourly, pred_col, args.out)
    fig_dawn_dusk(results, hourly, pred_col, solar, args.out)
    fig_novel_discoveries(results, hourly, pred_col, args.out, n=6)
    fig_confusion_matrix(results, pred_col, args.out, drop_crepuscular=True)
    fig_activity_workflow(results, hourly, pred_col, args.out, example_species=args.example_species)
    print("Done. Figures saved to:", args.out)

if __name__ == "__main__":
    main()

