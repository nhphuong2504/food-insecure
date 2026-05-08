"""Generate notebooks/01_eda_and_cleaning.ipynb."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_notebook import write_notebook  # noqa: E402

CELLS = [
    (
        "markdown",
        """# 01 - Exploratory Data Analysis and Cleaning

This notebook is the first step of the EE 5290 final project on household
food insecurity.  Its job is to:

1. load the provided FoodAPS-derived dataset and inspect the raw data
2. recode survey sentinel codes (`-996`, `-997`, `-998`) and string refusal
   codes (`"R"`) to NaN
3. add log-transformed distance features and a small number of household
   composition derivations
4. one-hot encode nominal categoricals and impute missing values
5. perform a stratified 80/20 train/test split and save the cleaned splits
   to `artifacts/` for the downstream prediction and segmentation notebooks
6. generate the figures that appear in the report's *Data and Preprocessing*
   section: class balance, marginal food-insecurity rates by key features,
   and a numeric correlation heatmap

All the heavy lifting lives in `src/data_prep.py`; this notebook only
exercises that module and produces narrative figures.""",
    ),
    (
        "code",
        """import os, sys
from pathlib import Path

# Allow imports from the project root regardless of where Jupyter is launched.
PROJECT_ROOT = Path.cwd()
if (PROJECT_ROOT / 'src').is_dir():
    pass
elif (PROJECT_ROOT.parent / 'src').is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

# Use a writable matplotlib cache (so the notebook works inside sandboxes).
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / '.cache' / 'matplotlib'))
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context='notebook', style='whitegrid', palette='deep')
plt.rcParams['figure.dpi'] = 100

FIG_DIR = PROJECT_ROOT / 'report' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

from src.data_prep import (
    OUTCOME, RANDOM_STATE,
    load_raw, clean, build_feature_matrix,
    make_train_test_split, save_artifacts,
)
""",
    ),
    (
        "markdown",
        """## 1. Raw data inspection

The raw data is one row per household with 38 columns covering food
security outcomes, economic conditions, household composition, employment
and demographics, food assistance program participation, transportation
and access indicators, and behavioral proxies.""",
    ),
    (
        "code",
        """raw = load_raw('data/Case_dataset.csv')
print('shape:', raw.shape)
print('positive rate (food_insecure_flag_adult):', raw[OUTCOME].mean().round(4))
raw.head(3)""",
    ),
    (
        "code",
        """# Map of column dtypes and overt missingness.
dtype_summary = pd.DataFrame({'dtype': raw.dtypes.astype(str), 'n_missing': raw.isna().sum()})
dtype_summary[dtype_summary['n_missing'] > 0]""",
    ),
    (
        "markdown",
        """### Sentinel codes

Several columns carry survey sentinel values such as `-996`, `-997`, and
`-998`.  These represent "refused", "don't know", and "not in universe"
responses, not real numeric values.  The following table makes the problem
concrete: in the raw data, the means of `caraccess`, `vehiclenum`, and
`anyvehicle` are dominated by these codes.""",
    ),
    (
        "code",
        """sentinel_columns = ['foodpantry', 'anyvehicle', 'vehiclenum', 'caraccess']
sentinel_summary = raw[sentinel_columns].describe().T[['min', 'max', 'mean']]
sentinel_summary['n_negative_codes'] = (raw[sentinel_columns] < 0).sum().values
sentinel_summary""",
    ),
    (
        "code",
        """# Drop and recode rationale (mirrored in src/data_prep.py):
print('caraccess sentinel share:', (raw['caraccess'] < 0).mean().round(4),
      '-> drop column entirely (>90% missing)')
print('fah_storetype_unique mean:', raw['fah_storetype_unique'].mean().round(6),
      '-> drop column (near-zero variance)')
print('hhwgt is the survey weight; not used as a feature.')""",
    ),
    (
        "markdown",
        """## 2. Cleaning, derivations, and feature matrix

`src.data_prep.clean` applies all sentinel and string-refusal recodings,
then `build_feature_matrix` drops leakage / ID / low-quality columns,
adds `log1p` distance features and household-composition derivations,
one-hot encodes the nominal categoricals, and median-imputes any
remaining numeric NaN.""",
    ),
    (
        "code",
        """cleaned = clean(raw)
print('post-clean shape:', cleaned.shape)
print()
print('residual NaN by column (after recoding sentinels and refusals):')
miss = cleaned.isna().sum()
miss[miss > 0]""",
    ),
    (
        "code",
        """fm = build_feature_matrix(cleaned)
print('X shape:', fm.X.shape)
print('y positive rate:', fm.y.mean().round(4))
print()
print('Feature group sizes:')
for grp, cols in fm.feature_groups.items():
    print(f'  {grp:>22s}: {len(cols):>3d} columns')""",
    ),
    (
        "markdown",
        """## 3. Stratified train/test split and persistence

We hold out 20% of the rows as a final test set, stratified on the
outcome.  All hyperparameter tuning happens via 5-fold stratified CV on
the training fold only; the test set is touched exactly once at the end
of the prediction notebook.""",
    ),
    (
        "code",
        """splits = make_train_test_split(fm, test_size=0.20, random_state=RANDOM_STATE)
for k, v in splits.items():
    print(f'{k}: {tuple(v.shape)} positive rate={float(v.mean() if k.startswith("y_") else v.mean().mean()):.4f}'
          if k.startswith('y_') else f'{k}: {tuple(v.shape)}')
save_artifacts(splits, out_dir='artifacts')
print('saved artifacts to artifacts/')""",
    ),
    (
        "markdown",
        """## 4. Class balance

Roughly 28% of the surveyed households are flagged as food insecure on the
adult food security category.  The imbalance is moderate, not extreme:
naive accuracy on the negative class would already exceed 70%, which
underscores the need to evaluate with PR-AUC and per-class recall rather
than raw accuracy.""",
    ),
    (
        "code",
        """fig, ax = plt.subplots(figsize=(5.5, 3.0))
counts = fm.y.value_counts().sort_index()
labels = ['Food secure (0)', 'Food insecure (1)']
bars = ax.bar(labels, counts.values, color=['#4C72B0', '#C44E52'])
for bar, n in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f'{n}\\n({n/counts.sum()*100:.1f}%)', ha='center', va='bottom')
ax.set_ylabel('Number of households')
ax.set_title('Class balance of food_insecure_flag_adult')
ax.set_ylim(0, counts.max() * 1.18)
sns.despine()
fig.tight_layout()
fig.savefig(FIG_DIR / '01_class_balance.png', dpi=200, bbox_inches='tight')
plt.show()""",
    ),
    (
        "markdown",
        """## 5. Marginal food-insecurity rates by selected features

These bar charts show the share of food-insecure households *within* each
level of selected features.  They are descriptive and intended to
motivate the modeling sections; they do not control for confounding.""",
    ),
    (
        "code",
        """def marginal_rate(df, group_col, outcome_col=OUTCOME, min_n=20):
    g = df.groupby(group_col)[outcome_col]
    out = pd.DataFrame({'rate': g.mean(), 'n': g.size()})
    return out[out['n'] >= min_n].sort_values('rate', ascending=False)

marg_panels = {
    'Poverty band': ('poverty_band', cleaned),
    'Region (1=NE,2=MW,3=S,4=W)': ('region', cleaned),
    'Rural indicator': ('rural', cleaned),
    'Any employed adult': ('any_employed_adult', cleaned),
    'SNAP participation': ('snap_any', cleaned),
    'Head education category (1=low..6=high)': ('head_educcat', cleaned),
}

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, (title, (col, df)) in zip(axes.ravel(), marg_panels.items()):
    rates = marginal_rate(df, col).reset_index()
    rates[col] = rates[col].astype(str)
    sns.barplot(data=rates, x=col, y='rate', ax=ax, color='#C44E52')
    ax.axhline(cleaned[OUTCOME].mean(), color='gray', linestyle='--', linewidth=1,
               label=f'Overall mean = {cleaned[OUTCOME].mean():.2f}')
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('Food-insecure rate')
    ax.set_xlabel('')
    ax.set_ylim(0, max(0.5, rates['rate'].max() * 1.1))
    ax.legend(loc='upper right', fontsize=8)
    for label in ax.get_xticklabels():
        label.set_rotation(20)
        label.set_ha('right')
fig.suptitle('Marginal food-insecure rate by selected features', y=1.02, fontsize=12)
fig.tight_layout()
fig.savefig(FIG_DIR / '02_marginal_rates.png', dpi=200, bbox_inches='tight')
plt.show()""",
    ),
    (
        "markdown",
        """## 6. Numeric correlation heatmap

Restricted to the continuous numeric features (distances, household
composition derivations, head age and education).  We expect distance
features to be strongly correlated with each other and with the rural
indicator.""",
    ),
    (
        "code",
        """numeric_cols = fm.numeric_columns
corr = fm.X[numeric_cols].corr()
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            annot=False, square=False, cbar_kws={'shrink': 0.7, 'label': 'Pearson r'})
ax.set_title('Correlation among continuous numeric features')
fig.tight_layout()
fig.savefig(FIG_DIR / '03_numeric_correlation.png', dpi=200, bbox_inches='tight')
plt.show()""",
    ),
    (
        "markdown",
        """## 7. Distance distribution check

Distances are right-skewed; we keep both the raw and `log1p` form so
linear models can fit a smooth signal while tree models can use the raw
units.""",
    ),
    (
        "code",
        """dist_cols = ['dist_sm', 'dist_walmart', 'nearsnap_dist', 'nearff_dist']
fig, axes = plt.subplots(2, 4, figsize=(14, 5.5))
for j, c in enumerate(dist_cols):
    sns.histplot(cleaned[c], ax=axes[0, j], bins=40, color='#4C72B0')
    axes[0, j].set_title(f'{c}')
    sns.histplot(cleaned[f'{c}_log1p'], ax=axes[1, j], bins=40, color='#55A868')
    axes[1, j].set_title(f'log1p({c})')
fig.suptitle('Raw vs. log1p distance distributions', y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / '04_distance_distributions.png', dpi=200, bbox_inches='tight')
plt.show()""",
    ),
    (
        "markdown",
        """## Outputs

The following artifacts have been written for downstream notebooks:

* `artifacts/X_train.csv`, `artifacts/X_test.csv` -- numeric feature matrices
* `artifacts/y_train.csv`, `artifacts/y_test.csv` -- binary outcome
* `report/figures/01_class_balance.png`
* `report/figures/02_marginal_rates.png`
* `report/figures/03_numeric_correlation.png`
* `report/figures/04_distance_distributions.png`""",
    ),
]


if __name__ == "__main__":
    write_notebook("notebooks/01_eda_and_cleaning.ipynb", CELLS)
    print("wrote notebooks/01_eda_and_cleaning.ipynb")
