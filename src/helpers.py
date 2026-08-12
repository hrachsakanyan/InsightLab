"""Plotting and statistics helpers for the InsightLab EDA.

Everything visual is centralised here so the notebook stays readable as an
analysis rather than as a pile of matplotlib boilerplate, and so every figure
comes out of the same validated palette.

Palette provenance
------------------
The categorical hues are assigned in a fixed slot order and were checked with a
colour-vision-deficiency validator against the light chart surface ``#fcfcfb``:

* 2 slots, all pairs  -> worst CVD dE 24.7, normal-vision dE 33.6  (pass)
* 4 slots, adjacent   -> worst CVD dE  9.1, normal-vision dE 22.9  (pass)

The 4-slot set sits below 3:1 contrast on aqua and yellow, so any chart using
all four ships direct value labels as relief. Sequential encodings use a single
blue ramp; the correlation matrix uses a blue<->red diverging ramp with a
neutral grey midpoint, so zero reads as "nothing".
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

#: Categorical slots, in fixed order. Never cycled, never reassigned by rank.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

#: Stable identity colours, so a hue means the same thing in every figure.
TARGET_COLORS = {"No disease": SERIES[0], "Disease": SERIES[1]}
SITE_COLORS = {
    "Cleveland": SERIES[0],
    "Hungary": SERIES[1],
    "Switzerland": SERIES[2],
    "Long Beach VA": SERIES[3],
}

#: Single-hue blue ramp (steps 100-700) for magnitude.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "insightlab_seq",
    ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)

#: Blue <-> red with a neutral grey midpoint, for signed quantities.
DIVERGING = LinearSegmentedColormap.from_list(
    "insightlab_div",
    ["#0d366b", "#2a78d6", "#9ec5f4", "#f0efec", "#f2a3a2", "#e34948", "#8f2020"],
)


def set_style() -> None:
    """Apply the project's chart theme: thin marks, hairline recessive chrome."""
    sns.set_theme(style="whitegrid")
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK_PRIMARY,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 10,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "text.color": INK_PRIMARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "figure.autolayout": False,
    })


def save_fig(fig: plt.Figure, name: str, directory: Path = FIGURES_DIR) -> Path:
    """Write a figure to ``figures/`` as PNG and return the path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path)
    return path


def _despine(ax: plt.Axes, keep_x: bool = True) -> None:
    """Drop the top/right spines and the vertical grid; keep chrome recessive."""
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_visible(keep_x)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)


# --------------------------------------------------------------------------
# Missingness
# --------------------------------------------------------------------------

def missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column missing counts and percentages, worst first."""
    n_missing = df.isna().sum()
    out = pd.DataFrame({
        "missing": n_missing,
        "missing_pct": (n_missing / len(df) * 100).round(1),
        "present": len(df) - n_missing,
    })
    return out.sort_values("missing", ascending=False)


def missingness_by_site(df: pd.DataFrame, site_col: str = "site") -> pd.DataFrame:
    """Missing percentage per column, broken out by collecting site."""
    return (
        df.groupby(site_col, observed=True)
        .apply(lambda g: g.isna().mean() * 100, include_groups=False)
        .round(1)
    )


def plot_missingness_by_site(df: pd.DataFrame, site_col: str = "site") -> plt.Figure:
    """Heatmap of missing-rate by column and site (sequential = magnitude)."""
    table = missingness_by_site(df, site_col)
    table = table.loc[:, table.max() > 0].sort_values(
        by=list(table.index), axis=1, ascending=False
    )

    fig, ax = plt.subplots(figsize=(11, 3.2))
    sns.heatmap(
        table, ax=ax, cmap=SEQUENTIAL, vmin=0, vmax=100,
        annot=True, fmt=".0f", annot_kws={"size": 9},
        linewidths=2, linecolor=SURFACE,
        cbar_kws={"label": "% missing", "shrink": 0.85, "pad": 0.02},
    )
    ax.set_title("Missing data is a property of the hospital, not of the variable")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(left=False, bottom=False)
    plt.setp(ax.get_yticklabels(), rotation=0, color=INK_SECONDARY)
    plt.setp(ax.get_xticklabels(), rotation=0, color=INK_SECONDARY)
    return fig


# --------------------------------------------------------------------------
# Univariate
# --------------------------------------------------------------------------

def numeric_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Describe-style summary with the quantities an analyst actually reads."""
    rows = []
    for col in cols:
        s = df[col].dropna()
        rows.append({
            "variable": col,
            "n": len(s),
            "missing": int(df[col].isna().sum()),
            "mean": s.mean(),
            "sd": s.std(),
            "min": s.min(),
            "q1": s.quantile(0.25),
            "median": s.median(),
            "q3": s.quantile(0.75),
            "max": s.max(),
            "skew": s.skew(),
        })
    return pd.DataFrame(rows).set_index("variable").round(2)


def plot_numeric_distributions(
    df: pd.DataFrame, cols: list[str], ncols: int = 3
) -> plt.Figure:
    """Histogram grid for the continuous variables — one series, one colour."""
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.1 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        s = df[col].dropna()
        ax.hist(s, bins=30, color=SERIES[0], edgecolor=SURFACE, linewidth=0.6)
        ax.axvline(s.median(), color=INK_PRIMARY, linewidth=1.5, linestyle="-")
        ax.annotate(
            f"median {s.median():g}",
            xy=(s.median(), ax.get_ylim()[1]), xytext=(4, -10),
            textcoords="offset points", fontsize=8, color=INK_SECONDARY,
            va="top",
        )
        ax.set_title(col)
        ax.set_ylabel("patients")
        _despine(ax)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle(
        "Distribution of the continuous measurements",
        x=0.005, ha="left", size=13, weight="600", color=INK_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def plot_categorical_counts(
    df: pd.DataFrame, cols: list[str], ncols: int = 3
) -> plt.Figure:
    """Count bars for the coded categoricals, with direct value labels."""
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        counts = df[col].value_counts(dropna=False).reindex(
            list(df[col].cat.categories) + ([np.nan] if df[col].isna().any() else [])
        )
        labels = ["(missing)" if pd.isna(i) else str(i) for i in counts.index]
        ax.barh(labels, counts.values, color=SERIES[0], height=0.62)
        ax.invert_yaxis()
        for y, v in enumerate(counts.values):
            ax.annotate(
                f"{int(v)}", xy=(v, y), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color=INK_SECONDARY,
            )
        ax.set_title(col)
        ax.set_xlabel("patients")
        ax.margins(x=0.16)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        ax.tick_params(left=False)
        plt.setp(ax.get_yticklabels(), color=INK_SECONDARY)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle(
        "Composition of the categorical variables",
        x=0.005, ha="left", size=13, weight="600", color=INK_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# --------------------------------------------------------------------------
# Proportions
# --------------------------------------------------------------------------

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Preferred over the normal approximation because several of the groups here
    are small, and Wald intervals misbehave badly near 0 and 1.
    """
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_by_category(
    df: pd.DataFrame, col: str, target: str = "disease"
) -> pd.DataFrame:
    """Disease rate within each level of ``col``, with Wilson intervals."""
    rows = []
    for level, group in df.groupby(col, observed=True):
        k, n = int(group[target].sum()), len(group)
        low, high = wilson_ci(k, n)
        rows.append({
            "level": str(level), "n": n, "cases": k,
            "rate": k / n if n else np.nan, "ci_low": low, "ci_high": high,
        })
    return pd.DataFrame(rows).set_index("level")


def plot_rate_by_category(
    df: pd.DataFrame, cols: list[str], target: str = "disease", ncols: int = 3
) -> plt.Figure:
    """Disease rate per category level, with 95% Wilson error bars.

    One measure, so one colour; the baseline rate is drawn as a reference line
    and every bar is directly labelled with its group size.
    """
    overall = df[target].mean()
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        table = rate_by_category(df, col, target)
        y = np.arange(len(table))
        err = np.vstack([
            table["rate"] - table["ci_low"],
            table["ci_high"] - table["rate"],
        ])
        ax.barh(y, table["rate"], color=SERIES[0], height=0.6)
        ax.errorbar(
            table["rate"], y, xerr=err, fmt="none",
            ecolor=INK_SECONDARY, elinewidth=1.2, capsize=3,
        )
        ax.axvline(overall, color=INK_MUTED, linewidth=1.2, zorder=0)
        for i, (rate, n) in enumerate(zip(table["rate"], table["n"])):
            ax.annotate(
                f"{rate:.0%}  (n={n})",
                xy=(table["ci_high"].iloc[i], i), xytext=(6, 0),
                textcoords="offset points", va="center",
                fontsize=8, color=INK_SECONDARY,
            )
        ax.set_yticks(y, table.index)
        ax.invert_yaxis()
        ax.set_xlim(0, 1.32)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_title(col)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)
        ax.tick_params(left=False)
        plt.setp(ax.get_yticklabels(), color=INK_SECONDARY)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Disease rate by category  ·  grey line = overall {overall:.0%}",
        x=0.005, ha="left", size=13, weight="600", color=INK_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# --------------------------------------------------------------------------
# Bivariate: numeric vs target
# --------------------------------------------------------------------------

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(a > b) - P(a < b).

    A rank-based effect size, so it pairs with Mann-Whitney and does not assume
    normality. Conventional reading: |d| < 0.147 negligible, < 0.33 small,
    < 0.474 medium, else large.
    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    # Rank-based identity, so this stays O(n log n) rather than O(n*m).
    ranks = stats.rankdata(np.concatenate([a, b]))
    rank_sum_a = ranks[: len(a)].sum()
    u_a = rank_sum_a - len(a) * (len(a) + 1) / 2
    return 2 * u_a / (len(a) * len(b)) - 1


def compare_numeric_by_target(
    df: pd.DataFrame, cols: list[str], target: str = "disease"
) -> pd.DataFrame:
    """Mann-Whitney U test plus Cliff's delta for each continuous variable.

    Mann-Whitney rather than a t-test because several of these distributions are
    skewed (``oldpeak`` especially); the effect size is reported alongside the
    p-value because with n~900 a tiny difference can still be "significant".
    """
    rows = []
    for col in cols:
        sub = df[[col, target]].dropna()
        pos = sub.loc[sub[target] == 1, col].to_numpy()
        neg = sub.loc[sub[target] == 0, col].to_numpy()
        u_stat, p = stats.mannwhitneyu(pos, neg, alternative="two-sided")
        delta = cliffs_delta(pos, neg)
        rows.append({
            "variable": col,
            "n_used": len(sub),
            "median_no_disease": np.median(neg),
            "median_disease": np.median(pos),
            "difference": np.median(pos) - np.median(neg),
            "cliffs_delta": delta,
            "effect": interpret_delta(delta),
            "p_value": p,
        })
    out = pd.DataFrame(rows).set_index("variable")
    return out.reindex(out["cliffs_delta"].abs().sort_values(ascending=False).index)


def interpret_delta(delta: float) -> str:
    """Label a Cliff's delta using the conventional thresholds."""
    d = abs(delta)
    if np.isnan(d):
        return "n/a"
    if d < 0.147:
        return "negligible"
    if d < 0.330:
        return "small"
    if d < 0.474:
        return "medium"
    return "large"


def plot_numeric_by_target(
    df: pd.DataFrame, cols: list[str], target: str = "disease", ncols: int = 3
) -> plt.Figure:
    """Overlaid densities per outcome group — two series, two fixed hues."""
    labels = {0: "No disease", 1: "Disease"}
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.1 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, col in zip(axes, cols):
        for code, label in labels.items():
            s = df.loc[df[target] == code, col].dropna()
            sns.kdeplot(
                x=s, ax=ax, color=TARGET_COLORS[label], fill=True,
                alpha=0.18, linewidth=2, label=label, cut=0,
            )
        ax.set_title(col)
        ax.set_ylabel("density")
        ax.set_xlabel("")
        ax.legend(loc="upper right")
        _despine(ax)

    for ax in axes[len(cols):]:
        ax.set_visible(False)

    fig.suptitle(
        "Continuous measurements, split by outcome",
        x=0.005, ha="left", size=13, weight="600", color=INK_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


# --------------------------------------------------------------------------
# Association / correlation
# --------------------------------------------------------------------------

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramer's V for the association between two categoricals.

    Uses the Bergsma-Wicher correction, which matters here because some cells
    are thin once the data is split by site.
    """
    pair = pd.DataFrame({"x": x, "y": y}).dropna()
    if pair["x"].nunique() < 2 or pair["y"].nunique() < 2:
        return np.nan
    table = pd.crosstab(pair["x"], pair["y"])
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else np.nan


def spearman_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Pairwise Spearman correlations.

    Spearman rather than Pearson: ``oldpeak`` and ``chol`` are skewed and the
    relationships of interest are monotone rather than strictly linear.
    """
    return df[cols].corr(method="spearman")


def plot_correlation_heatmap(
    corr: pd.DataFrame, title: str = "Spearman correlation"
) -> plt.Figure:
    """Lower-triangle correlation heatmap on a diverging blue<->red ramp."""
    # Mask the diagonal too: a row of 1.00s is the darkest thing on the plot and
    # carries no information. Tick labels still anchor each row and column.
    mask = np.triu(np.ones_like(corr, dtype=bool), k=0)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    sns.heatmap(
        corr, mask=mask, ax=ax, cmap=DIVERGING, vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 9},
        linewidths=2, linecolor=SURFACE, square=True,
        cbar_kws={"label": "rho", "shrink": 0.7, "pad": 0.02},
    )
    ax.set_title(title)
    ax.grid(False)  # otherwise the theme grid shows through the masked triangle
    ax.tick_params(left=False, bottom=False)
    plt.setp(ax.get_yticklabels(), rotation=0, color=INK_SECONDARY)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", color=INK_SECONDARY)
    return fig


def target_association_ranking(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target: str = "disease",
) -> pd.DataFrame:
    """Rank every variable by strength of association with the target.

    Continuous variables are scored with |Cliff's delta| and categoricals with
    Cramer's V. Both live on 0-1 and both are rank/contingency based, so they
    are comparable enough to sort into one table — which is what an analyst
    wants before choosing features.
    """
    rows = []
    for col in numeric_cols:
        sub = df[[col, target]].dropna()
        delta = cliffs_delta(
            sub.loc[sub[target] == 1, col].to_numpy(),
            sub.loc[sub[target] == 0, col].to_numpy(),
        )
        rows.append({
            "variable": col, "kind": "continuous",
            "strength": abs(delta), "measure": "|Cliff's delta|",
            "n_used": len(sub),
        })
    for col in categorical_cols:
        sub = df[[col, target]].dropna()
        rows.append({
            "variable": col, "kind": "categorical",
            "strength": cramers_v(sub[col], sub[target]), "measure": "Cramer's V",
            "n_used": len(sub),
        })
    out = pd.DataFrame(rows).set_index("variable")
    return out.sort_values("strength", ascending=False)


def plot_categorical_association_matrix(matrix: pd.DataFrame) -> plt.Figure:
    """Cramer's V between categoricals. Unsigned 0-1, so a sequential ramp."""
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=0)
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    sns.heatmap(
        matrix, mask=mask, ax=ax, cmap=SEQUENTIAL, vmin=0, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 9},
        linewidths=2, linecolor=SURFACE, square=True,
        cbar_kws={"label": "Cramer's V", "shrink": 0.7, "pad": 0.02},
    )
    ax.set_title("Association between the categorical variables")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(False)
    ax.tick_params(left=False, bottom=False)
    plt.setp(ax.get_yticklabels(), rotation=0, color=INK_SECONDARY)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", color=INK_SECONDARY)
    return fig


# --------------------------------------------------------------------------
# Bespoke figures for the findings that carry the analysis
# --------------------------------------------------------------------------

def plot_target_by_site(df: pd.DataFrame, target: str = "disease") -> plt.Figure:
    """Disease rate per hospital. One measure, so one colour plus direct labels."""
    table = rate_by_category(df, "site", target)
    overall = df[target].mean()

    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    y = np.arange(len(table))
    err = np.vstack([
        table["rate"] - table["ci_low"], table["ci_high"] - table["rate"],
    ])
    ax.barh(y, table["rate"], color=SERIES[0], height=0.58)
    ax.errorbar(
        table["rate"], y, xerr=err, fmt="none",
        ecolor=INK_SECONDARY, elinewidth=1.2, capsize=3,
    )
    ax.axvline(overall, color=INK_MUTED, linewidth=1.2, zorder=0)
    for i, row in enumerate(table.itertuples()):
        ax.annotate(
            f"{row.rate:.0%}   ({row.cases}/{row.n})",
            xy=(row.ci_high, i), xytext=(7, 0), textcoords="offset points",
            va="center", fontsize=9, color=INK_SECONDARY,
        )
    ax.set_yticks(y, table.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title(f"Disease rate varies 2.6x across hospitals  ·  overall {overall:.0%}")
    ax.set_xlabel("share of patients with >50% vessel narrowing")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.tick_params(left=False)
    plt.setp(ax.get_yticklabels(), color=INK_SECONDARY)
    fig.tight_layout()
    return fig


def plot_cholesterol_sign_flip(
    raw: pd.DataFrame, clean: pd.DataFrame
) -> plt.Figure:
    """Side-by-side of the cholesterol comparison before and after cleaning.

    The left panel keeps ``chol == 0`` as if it were a measurement; the right
    panel treats it as missing. The conclusion reverses sign, which is the
    single most important thing this notebook has to say.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    panels = [
        (axes[0], raw, "Before cleaning: chol = 0 kept as a value"),
        (axes[1], clean, "After cleaning: chol = 0 treated as missing"),
    ]

    for ax, frame, title in panels:
        sub = frame[["chol", "disease"]].dropna()
        bins = np.linspace(0, 620, 63)
        for code, label in [(0, "No disease"), (1, "Disease")]:
            ax.hist(
                sub.loc[sub.disease == code, "chol"], bins=bins,
                color=TARGET_COLORS[label], alpha=0.55, label=label,
                edgecolor=SURFACE, linewidth=0.5,
            )
        delta = cliffs_delta(
            sub.loc[sub.disease == 1, "chol"].to_numpy(),
            sub.loc[sub.disease == 0, "chol"].to_numpy(),
        )
        med_no = sub.loc[sub.disease == 0, "chol"].median()
        med_yes = sub.loc[sub.disease == 1, "chol"].median()
        direction = "higher" if delta > 0 else "lower"
        ax.set_title(title)
        ax.set_xlabel("serum cholesterol (mg/dl)")
        ax.annotate(
            f"Cliff's delta = {delta:+.3f}\n"
            f"median: {med_no:.0f} vs {med_yes:.0f} mg/dl\n"
            f"-> disease group reads {direction}\n"
            f"n = {len(sub)}",
            xy=(0.97, 0.72), xycoords="axes fraction", ha="right", va="top",
            fontsize=9, color=INK_SECONDARY,
        )
        ax.legend(loc="upper right")
        _despine(ax)

    axes[0].set_ylabel("patients")
    fig.suptitle(
        "One cleaning decision reverses the cholesterol finding",
        x=0.005, ha="left", size=13, weight="600", color=INK_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_effect_within_bands(
    df: pd.DataFrame,
    value: str,
    band: str,
    target: str = "disease",
) -> plt.Figure:
    """Median of ``value`` per outcome group within each band, plus effect size.

    This is the shape of a confounding check: if the gap between the two hues
    survives inside every band, the band variable does not explain it away.
    """
    sub = df[[value, band, target]].dropna()
    bands = list(sub[band].cat.categories) if hasattr(sub[band], "cat") \
        else sorted(sub[band].unique())

    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    x = np.arange(len(bands))
    width = 0.36

    for offset, (code, label) in zip((-width / 2, width / 2),
                                     [(0, "No disease"), (1, "Disease")]):
        medians, counts = [], []
        for b in bands:
            g = sub[(sub[band] == b) & (sub[target] == code)][value]
            medians.append(g.median())
            counts.append(len(g))
        bars = ax.bar(
            x + offset, medians, width - 0.02,  # 2px-equivalent surface gap
            color=TARGET_COLORS[label], label=label,
        )
        for bar, med, n in zip(bars, medians, counts):
            ax.annotate(
                f"{med:.0f}\nn={n}",
                xy=(bar.get_x() + bar.get_width() / 2, med), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=8,
                color=INK_SECONDARY,
            )

    # Effect size per band, pinned just under the top of the axes: x follows the
    # group, y is an axes fraction so the label never collides with a bar.
    from matplotlib.transforms import blended_transform_factory

    blend = blended_transform_factory(ax.transData, ax.transAxes)
    for i, b in enumerate(bands):
        g = sub[sub[band] == b]
        d = cliffs_delta(
            g.loc[g[target] == 1, value].to_numpy(),
            g.loc[g[target] == 0, value].to_numpy(),
        )
        ax.text(
            i, 0.035, f"delta {d:+.2f}\n{interpret_delta(d)}",
            transform=blend, ha="center", va="bottom",
            fontsize=8.5, color=INK_PRIMARY, weight="600",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE,
                      edgecolor=GRIDLINE, linewidth=0.8),
        )

    ax.set_xticks(x, bands)
    ax.set_xlabel(band)
    ax.set_ylabel(f"median {value}")
    ax.set_title(f"The {value} gap holds inside every {band}")
    ax.legend(loc="upper right", ncols=2)
    ax.margins(y=0.18)
    _despine(ax)
    plt.setp(ax.get_xticklabels(), color=INK_SECONDARY)
    fig.tight_layout()
    return fig


def plot_completeness_cascade(policies: dict[str, pd.DataFrame]) -> plt.Figure:
    """Site composition surviving each missing-data policy.

    Four sites as stacked identity segments (validated adjacent), each segment
    directly labelled so the sub-3:1 slots carry visible relief.
    """
    sites = list(SITE_COLORS)
    fig, ax = plt.subplots(figsize=(9.6, 0.85 * len(policies) + 1.8))
    y = np.arange(len(policies))

    lefts = np.zeros(len(policies))
    for site in sites:
        widths = np.array([
            int((frame["site"] == site).sum()) for frame in policies.values()
        ])
        ax.barh(
            y, widths, left=lefts, height=0.58,
            color=SITE_COLORS[site], label=site,
            edgecolor=SURFACE, linewidth=2,  # 2px surface gap between fills
        )
        for i, (w, l) in enumerate(zip(widths, lefts)):
            if w >= 40:  # only label where it fits with padding
                ax.annotate(
                    str(w), xy=(l + w / 2, i), ha="center", va="center",
                    fontsize=9, color=SURFACE, weight="600",
                )
        lefts += widths

    for i, total in enumerate(lefts):
        ax.annotate(
            f"{int(total)} rows",
            xy=(total, i), xytext=(8, 0), textcoords="offset points",
            va="center", fontsize=9, color=INK_SECONDARY,
        )

    ax.set_yticks(y, list(policies))
    ax.invert_yaxis()
    ax.set_xlabel("patients retained")
    ax.set_xlim(0, lefts.max() * 1.18)
    ax.set_title("Dropping incomplete rows quietly turns this into a Cleveland-only study")
    ax.legend(loc="lower right", ncols=4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.tick_params(left=False)
    plt.setp(ax.get_yticklabels(), color=INK_SECONDARY)
    fig.tight_layout()
    return fig


def plot_association_ranking(
    ranking: pd.DataFrame, target_label: str = "disease"
) -> plt.Figure:
    """Ranked association strengths, with the coverage caveat labelled inline."""
    data = ranking.dropna(subset=["strength"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(data) + 1.6))
    y = np.arange(len(data))
    ax.barh(y, data["strength"], color=SERIES[0], height=0.6)
    for i, (strength, n) in enumerate(zip(data["strength"], data["n_used"])):
        ax.annotate(
            f"{strength:.2f}   n={n}",
            xy=(strength, i), xytext=(6, 0), textcoords="offset points",
            va="center", fontsize=8, color=INK_SECONDARY,
        )
    ax.set_yticks(y, data.index)
    ax.set_xlim(0, max(data["strength"]) * 1.35)
    ax.set_xlabel("association strength  (|Cliff's delta| or Cramer's V)")
    ax.set_title(f"What actually tracks {target_label}?")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.xaxis.grid(True)
    ax.yaxis.grid(False)
    ax.tick_params(left=False)
    plt.setp(ax.get_yticklabels(), color=INK_SECONDARY)
    fig.tight_layout()
    return fig
