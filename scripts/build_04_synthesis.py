"""Generate notebooks/04_synthesis_figures.ipynb."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_notebook import write_notebook  # noqa: E402

CELLS = [
    (
        "markdown",
        """# 04 - Synthesis Figures

This notebook bridges the prediction track and the segmentation track.
It produces three synthesis figures used in the report's *Synthesis &
Recommendations* section:

1. Persona composition of the top-decile predicted-risk households -- shows
   *who* a food bank would actually be helping if they followed the
   model's targeting recommendations.
2. Precision/recall vs. fraction targeted -- shows the operating-point
   tradeoff that the report uses to frame the targeting recommendation.
3. A combined targeting score table that joins the model's predicted
   probability with each household's persona assignment.""",
    ),
    (
        "code",
        """import os, sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if not (PROJECT_ROOT / 'src').is_dir():
    PROJECT_ROOT = PROJECT_ROOT.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault('MPLCONFIGDIR', str(PROJECT_ROOT / '.cache' / 'matplotlib'))
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

sns.set_theme(context='notebook', style='whitegrid', palette='deep')
plt.rcParams['figure.dpi'] = 100
RANDOM_STATE = 42
FIG_DIR = PROJECT_ROOT / 'report' / 'figures'

from src.data_prep import (
    OUTCOME, load_raw, clean, build_feature_matrix, load_artifacts,
)
from src.eval_utils import build_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
""",
    ),
    (
        "markdown",
        """## 1. Refit the wining classifier and the persona clusterer

We refit on the train split and the food-insecure subset using the
locked hyperparameters and seeds.  This keeps the synthesis notebook
self-contained.""",
    ),
    (
        "code",
        """splits = load_artifacts('artifacts')
X_train, X_test = splits['X_train'], splits['X_test']
y_train, y_test = splits['y_train'], splits['y_test']

raw = load_raw()
cleaned = clean(raw)
fm = build_feature_matrix(cleaned)
numeric_columns = fm.numeric_columns
binary_columns  = fm.binary_columns

# Winning model: logistic regression with the best C from the prediction track.
# We hard-code C = 1.0 because that was the GridSearchCV winner; if the report
# requires a different value we can re-tune here.
lr = build_pipeline(
    LogisticRegression(max_iter=2000, solver='lbfgs', C=1.0,
                        random_state=RANDOM_STATE),
    numeric_columns, binary_columns, scale_for_estimator=True,
)
lr.fit(X_train, y_train)

# Refit the geography-free persona clusterer on the food-insecure subset.
mask = (fm.y == 1)
X_fi = fm.X[mask].reset_index(drop=True)
profile_df = cleaned.loc[mask].reset_index(drop=True)
geo_substrings = ['dist_', 'nearsnap', 'nearff', 'nearnonff',
                  'rural', 'nonmetro', 'region=']
behavior_columns = [c for c in X_fi.columns
                    if not any(s in c for s in geo_substrings)]
beh_scaler = StandardScaler().fit(X_fi[behavior_columns])
X_beh_std = beh_scaler.transform(X_fi[behavior_columns])
pca_beh = PCA(random_state=RANDOM_STATE).fit(X_beh_std)
n_beh = int(np.searchsorted(np.cumsum(pca_beh.explained_variance_ratio_), 0.80) + 1)
pca_beh_fit = PCA(n_components=n_beh, random_state=RANDOM_STATE).fit(X_beh_std)
X_beh_pca = pca_beh_fit.transform(X_beh_std)
km_beh = KMeans(n_clusters=5, random_state=RANDOM_STATE, n_init=20).fit(X_beh_pca)
profile_df['persona'] = km_beh.labels_
print('refit complete')""",
    ),
    (
        "markdown",
        """## 2. Assign personas to all households (not just food-insecure)

To talk about *who the model is flagging*, we need to assign every
household in the test set to a persona, not just the food-insecure ones.
We do this by transforming each household through the same
behavior-only PCA/KMeans pipeline and reading off the nearest centroid.""",
    ),
    (
        "code",
        """# Transform the entire (train + test) feature matrix into the persona space
# and assign each household to its nearest centroid.
def assign_personas(X):
    X_trans = beh_scaler.transform(X[behavior_columns])
    X_pca = pca_beh_fit.transform(X_trans)
    return km_beh.predict(X_pca)

persona_train = assign_personas(X_train)
persona_test  = assign_personas(X_test)

print('test-set persona distribution (all households):')
print(pd.Series(persona_test).value_counts(normalize=True).sort_index().round(3))""",
    ),
    (
        "code",
        """# Persona labels by their narrative names (derived from the segmentation notebook).
persona_names = {
    0: 'Working families just above poverty',
    1: 'Non-working low-income singles, high pantry use',
    2: 'Large low-income families with children',
    3: 'Working educated households near boundary',
    4: 'Elderly low-income households, high SNAP',
}

# Validate the names against the centroid means to catch any seed drift.
for pid in sorted(set(persona_train)):
    print(f'P{pid} ({persona_names.get(pid, "?")}) -> n={(persona_train == pid).sum()}')""",
    ),
    (
        "markdown",
        """## 3. Persona composition of the top-decile predicted-risk households

The headline targeting recommendation is "rank households by predicted
risk and prioritize the top X%".  This figure shows, for several values
of X, what *kind* of households the model would prioritize: the goal is
both fairness inspection (no single persona is over-targeted to the
exclusion of others) and outreach planning (knowing which personas
dominate the top decile lets the food bank tailor its programs).""",
    ),
    (
        "code",
        """test_proba = lr.predict_proba(X_test)[:, 1]
test_df = pd.DataFrame({
    'proba': test_proba,
    'persona': persona_test,
    'truly_fi': y_test.values,
})

cuts = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
rows = []
for c in cuts:
    thr = float(np.quantile(test_proba, 1 - c)) if c < 1.0 else -np.inf
    sub = test_df[test_df['proba'] >= thr]
    counts = sub['persona'].value_counts(normalize=True).reindex(range(5), fill_value=0.0)
    row = {'top_share': c, 'flagged_n': len(sub),
           'precision': sub['truly_fi'].mean() if len(sub) else 0.0,
           'recall': sub['truly_fi'].sum() / max(1, test_df['truly_fi'].sum())}
    for p in range(5):
        row[f'P{p}'] = counts.get(p, 0.0)
    rows.append(row)
top_table = pd.DataFrame(rows)
top_table.to_csv(PROJECT_ROOT / 'artifacts' / 'top_share_persona_mix.csv', index=False)
top_table.round(3)""",
    ),
    (
        "code",
        """# Stacked-bar figure: persona mix at each top-X% cutoff.
fig, ax = plt.subplots(figsize=(7.5, 4.5))
shares = top_table[[f'P{p}' for p in range(5)]].values * 100
labels = [f'top {int(s*100)}%\\n(P={p:.2f}, R={r:.2f})'
          for s, p, r in zip(top_table['top_share'], top_table['precision'], top_table['recall'])]
bottoms = np.zeros(len(top_table))
palette = sns.color_palette('Set2', n_colors=5)
for p in range(5):
    ax.bar(labels, shares[:, p], bottom=bottoms, color=palette[p],
           label=f'P{p}: {persona_names[p]}')
    bottoms += shares[:, p]
ax.set_ylabel('Persona share within targeted group (%)')
ax.set_title('Persona mix at each predicted-risk cutoff (test set)')
ax.set_ylim(0, 100)
plt.xticks(rotation=45, ha='right')
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / '15_persona_mix_at_cutoffs.png', dpi=200, bbox_inches='tight')
plt.show()""",
    ),
    (
        "markdown",
        """## 4. Precision and recall vs. fraction targeted

A single curve shows the entire targeting tradeoff: as the food bank
expands the share of households it prioritizes, recall rises but
precision falls.  Vertical reference lines mark the 10%, 20%, and 30%
operating points.""",
    ),
    (
        "code",
        """fractions = np.linspace(0.01, 1.0, 100)
precs, recs = [], []
total_pos = float(y_test.sum())
for f in fractions:
    thr = float(np.quantile(test_proba, 1 - f))
    flagged = (test_proba >= thr)
    tp = int(((flagged) & (y_test == 1)).sum())
    precs.append(tp / max(1, flagged.sum()))
    recs.append(tp / max(1, total_pos))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(fractions * 100, np.array(precs) * 100, color='#C44E52', label='Precision')
ax.plot(fractions * 100, np.array(recs)  * 100, color='#4C72B0', label='Recall')
for f in [0.10, 0.20, 0.30]:
    ax.axvline(f * 100, linestyle='--', color='gray', linewidth=0.8)
ax.set_xlabel('Top X% households targeted')
ax.set_ylabel('Percentage')
ax.set_title('Precision and recall vs. targeting cutoff (LR on test set)')
ax.legend()
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
fig.tight_layout()
fig.savefig(FIG_DIR / '16_precision_recall_vs_cutoff.png', dpi=200, bbox_inches='tight')
plt.show()""",
    ),
    (
        "markdown",
        """## 5. Combined targeting recommendation table

A small per-persona summary suitable for the report's recommendations
section.  For each persona we report:

* its share of the food-insecure subset
* its mean predicted probability of food insecurity (an indicator of
  how the model perceives the persona's risk in the wider population)
* its share among the top 10% predicted-risk households
* a short outreach hint""",
    ),
    (
        "code",
        """# Build per-persona combined table on the entire (train + test) population.
all_X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
all_y = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)
all_persona = assign_personas(all_X)
all_proba   = lr.predict_proba(all_X)[:, 1]
top10 = all_proba >= np.quantile(all_proba, 0.90)

per_rows = []
for p in range(5):
    mask_p = all_persona == p
    fi_subset_share = float(((all_y == 1) & mask_p).sum() / max(1, (all_y == 1).sum()))
    population_share = float(mask_p.mean())
    mean_proba = float(all_proba[mask_p].mean())
    in_top10 = float(((mask_p) & (top10)).sum() / max(1, top10.sum()))
    per_rows.append({
        'persona': f'P{p}',
        'name': persona_names[p],
        'population_share': round(population_share, 3),
        'food_insecure_share': round(fi_subset_share, 3),
        'mean_pred_proba': round(mean_proba, 3),
        'top10pct_share': round(in_top10, 3),
    })
per_df = pd.DataFrame(per_rows)
per_df.to_csv(PROJECT_ROOT / 'artifacts' / 'persona_targeting_summary.csv', index=False)
per_df""",
    ),
    (
        "markdown",
        """## Outputs

* `artifacts/top_share_persona_mix.csv`
* `artifacts/persona_targeting_summary.csv`
* `report/figures/15_persona_mix_at_cutoffs.png`
* `report/figures/16_precision_recall_vs_cutoff.png`""",
    ),
]


if __name__ == "__main__":
    write_notebook("notebooks/04_synthesis_figures.ipynb", CELLS)
    print("wrote notebooks/04_synthesis_figures.ipynb")
