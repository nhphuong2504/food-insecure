"""Generate notebooks/02_prediction_track.ipynb."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_notebook import write_notebook  # noqa: E402

CELLS = [
    (
        "markdown",
        """# 02 - Track 1: Predictive Modeling

This notebook is the predictive-modeling track of the EE 5290 final
project.  It compares **four classifiers** -- logistic regression,
linear-kernel SVM, RBF-kernel SVM, and random forest -- under three
class-imbalance strategies (no adjustment, balanced class weights,
F1-tuned threshold).

The output is a **12-row** metrics table on the held-out test set,
plus PR curves, calibration diagnostics, and feature-importance
summaries that feed the report.

All hyperparameter tuning happens via 5-fold stratified CV on the
**training fold only**; the test set is touched exactly once at the end.""",
    ),
    (
        "code",
        """import os, sys, time
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

from src.data_prep import load_raw, clean, build_feature_matrix, load_artifacts
from src.eval_utils import (
    build_pipeline, tune_threshold, compute_test_metrics,
    plot_pr_curves, plot_calibration, recall_at_precision,
)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
""",
    ),
    (
        "code",
        """splits = load_artifacts('artifacts')
X_train, X_test = splits['X_train'], splits['X_test']
y_train, y_test = splits['y_train'], splits['y_test']

# Recover numeric / binary column lists from the same data prep used to build
# the splits so that build_pipeline can decide what to standardize.
fm = build_feature_matrix(clean(load_raw()))
numeric_columns = fm.numeric_columns
binary_columns  = fm.binary_columns
print('train shape:', X_train.shape, 'test shape:', X_test.shape)
print('positive rate (train):', round(y_train.mean(), 4))
print('# numeric (scaled) columns:', len(numeric_columns))
print('# binary / one-hot columns:', len(binary_columns))""",
    ),
    (
        "markdown",
        """## 1. Cross-validated grid search

We tune three estimators with small grids using 5-fold stratified CV.
The optimization metric is **average precision** (PR-AUC) which is the
sensible choice for imbalanced binary classification: it ignores the
abundant true negatives and focuses on the precision/recall tradeoff
across thresholds.""",
    ),
    (
        "code",
        """def fit_search(name, estimator, scale, param_grid):
    pipe = build_pipeline(estimator, numeric_columns, binary_columns,
                          scale_for_estimator=scale)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        pipe, param_grid=param_grid, scoring='average_precision',
        cv=cv, n_jobs=-1, refit=True, verbose=0,
    )
    t0 = time.time()
    gs.fit(X_train, y_train)
    dt = time.time() - t0
    print(f'{name:<28s} best PR-AUC = {gs.best_score_:.4f}  '
          f'best params = {gs.best_params_}  ({dt:.1f}s)')
    return gs

# Strategy A: no imbalance adjustment
search_lr_none = fit_search(
    'LR (no adjust)',
    LogisticRegression(max_iter=2000, solver='lbfgs', random_state=RANDOM_STATE),
    scale=True,
    param_grid={'clf__C': [0.01, 0.1, 1.0, 10.0]},
)
search_svm_lin_none = fit_search(
    'SVM-Linear (no adjust)',
    SVC(kernel='linear', probability=True, random_state=RANDOM_STATE,
        cache_size=500),
    scale=True,
    param_grid={'clf__C': [0.01, 0.1, 1.0, 10.0]},
)
search_svm_none = fit_search(
    'SVM-RBF (no adjust)',
    SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE, cache_size=500),
    scale=True,
    param_grid={'clf__C': [0.5, 1.0, 5.0], 'clf__gamma': ['scale', 0.01, 0.1]},
)
search_rf_none = fit_search(
    'RF (no adjust)',
    RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    scale=False,
    param_grid={'clf__max_depth': [None, 10, 20], 'clf__min_samples_leaf': [1, 5]},
)""",
    ),
    (
        "code",
        """# Strategy B: balanced class weights.  We reuse the best hyperparameters from
# Strategy A's CV (a common pragmatic shortcut that keeps total compute
# reasonable; the imbalance comparison is honest because the threshold
# tuning is done independently per (model, strategy)).
def best_clf(search):
    return search.best_estimator_.named_steps['clf']

def refit_with(estimator, scale):
    pipe = build_pipeline(estimator, numeric_columns, binary_columns,
                          scale_for_estimator=scale)
    pipe.fit(X_train, y_train)
    return pipe

lr_balanced = refit_with(
    LogisticRegression(max_iter=2000, solver='lbfgs',
                        C=best_clf(search_lr_none).C,
                        class_weight='balanced',
                        random_state=RANDOM_STATE),
    scale=True,
)
svm_lin_balanced = refit_with(
    SVC(kernel='linear', probability=True,
        C=best_clf(search_svm_lin_none).C,
        class_weight='balanced',
        random_state=RANDOM_STATE, cache_size=500),
    scale=True,
)
svm_balanced = refit_with(
    SVC(kernel='rbf', probability=True,
        C=best_clf(search_svm_none).C,
        gamma=best_clf(search_svm_none).gamma,
        class_weight='balanced',
        random_state=RANDOM_STATE, cache_size=500),
    scale=True,
)
rf_balanced = refit_with(
    RandomForestClassifier(n_estimators=300,
                            max_depth=best_clf(search_rf_none).max_depth,
                            min_samples_leaf=best_clf(search_rf_none).min_samples_leaf,
                            class_weight='balanced',
                            random_state=RANDOM_STATE, n_jobs=-1),
    scale=False,
)
print('balanced refits done')""",
    ),
    (
        "markdown",
        """## 2. Predict probabilities and tune thresholds

For each (model, strategy) we record:

* the probability vector on the held-out test set
* a **default-threshold** (0.5) row -- the unadjusted classifier
* a **F1-tuned threshold** row -- threshold chosen on the *training fold's*
  out-of-fold predictions to avoid touching the test set

Threshold tuning uses 5-fold OOF predictions of the chosen model so the
threshold itself never sees the test labels.""",
    ),
    (
        "code",
        """from sklearn.model_selection import cross_val_predict

def get_oof_proba(pipe):
    return cross_val_predict(
        pipe, X_train, y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        method='predict_proba', n_jobs=-1,
    )[:, 1]

models = {
    'LR (none)':           search_lr_none.best_estimator_,
    'LR (balanced)':       lr_balanced,
    'SVM-Lin (none)':      search_svm_lin_none.best_estimator_,
    'SVM-Lin (balanced)':  svm_lin_balanced,
    'SVM-RBF (none)':      search_svm_none.best_estimator_,
    'SVM-RBF (balanced)':  svm_balanced,
    'RF (none)':           search_rf_none.best_estimator_,
    'RF (balanced)':       rf_balanced,
}

# Get test probabilities once.
test_proba = {name: pipe.predict_proba(X_test)[:, 1] for name, pipe in models.items()}

# OOF probabilities for fair threshold tuning.
print('Generating out-of-fold predictions for threshold tuning...')
oof_proba = {}
for name, pipe in models.items():
    t0 = time.time()
    oof_proba[name] = get_oof_proba(pipe)
    print(f'  {name:<22s} ({time.time()-t0:.1f}s)')""",
    ),
    (
        "code",
        """tuned_thr = {name: tune_threshold(y_train, oof_proba[name], 'f1')
              for name in models}
print('F1-tuned thresholds (chosen on training OOF predictions):')
for name, thr in tuned_thr.items():
    print(f'  {name:<22s} thr = {thr:.3f}')""",
    ),
    (
        "markdown",
        """## 3. Build the 12-row results table

For each base model (LR, SVM-Linear, SVM-RBF, RF) we report three rows:
no adjustment at default threshold, balanced class weights at default
threshold, and threshold-tuned (using the **base** model, since adding
`class_weight` produces a similar but redundant variant in our experience).

The locked primary metrics are PR-AUC, F1, recall, and recall at fixed
precision floors (0.50 and 0.70).""",
    ),
    (
        "code",
        """rows = []
for base in ['LR', 'SVM-Lin', 'SVM-RBF', 'RF']:
    none_name = f'{base} (none)'
    bal_name  = f'{base} (balanced)'
    rows.append(compute_test_metrics(
        f'{base} - no adjust (thr=0.50)',
        y_test, test_proba[none_name], threshold=0.50).as_row())
    rows.append(compute_test_metrics(
        f'{base} - balanced (thr=0.50)',
        y_test, test_proba[bal_name],  threshold=0.50).as_row())
    rows.append(compute_test_metrics(
        f'{base} - F1-tuned thr',
        y_test, test_proba[none_name], threshold=tuned_thr[none_name]).as_row())

results = pd.DataFrame(rows)
results""",
    ),
    (
        "code",
        """results.to_csv(PROJECT_ROOT / 'artifacts' / 'prediction_results_12row.csv', index=False)
# Keep a 9-row alias for any downstream consumer that expected the old name.
results.to_csv(PROJECT_ROOT / 'artifacts' / 'prediction_results_9row.csv', index=False)
print('saved artifacts/prediction_results_12row.csv')""",
    ),
    (
        "markdown",
        """## 4. Visual comparison: PR curves

A single panel showing the precision/recall curves of the four base
classifiers (no-adjust variants), against the random-baseline horizontal
line at the test-set positive rate.  Threshold tuning corresponds to
picking a particular point on each curve; class weights bend the curves
only modestly, so we omit those for visual clarity (they are already in
the table).""",
    ),
    (
        "code",
        """fig, ax = plt.subplots(figsize=(6.5, 4.8))
plot_pr_curves(
    [
        ('Logistic Regression',   y_test, test_proba['LR (none)']),
        ('SVM (linear kernel)',   y_test, test_proba['SVM-Lin (none)']),
        ('SVM (RBF kernel)',      y_test, test_proba['SVM-RBF (none)']),
        ('Random Forest',         y_test, test_proba['RF (none)']),
    ],
    ax=ax,
    title='Test-set Precision-Recall curves',
    baseline_positive_rate=float(y_test.mean()),
)
fig.tight_layout()
fig.savefig(FIG_DIR / '05_pr_curves.png', dpi=200, bbox_inches='tight')
plt.show()""",
    ),
    (
        "markdown",
        """## 5. Calibration diagnostics

Reliability diagrams show whether the predicted probability of food
insecurity matches the empirical positive rate within probability bins.
Logistic regression is typically well calibrated; tree ensembles less so.""",
    ),
    (
        "code",
        """fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), sharey=True)
plot_calibration(y_test, test_proba['LR (none)'],      ax=axes[0], name='LR')
plot_calibration(y_test, test_proba['SVM-Lin (none)'], ax=axes[1], name='SVM-Lin')
plot_calibration(y_test, test_proba['SVM-RBF (none)'], ax=axes[2], name='SVM-RBF')
plot_calibration(y_test, test_proba['RF (none)'],      ax=axes[3], name='RF')
fig.suptitle('Test-set calibration (quantile-binned)', y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / '06_calibration.png', dpi=200, bbox_inches='tight')
plt.show()""",
    ),
    (
        "markdown",
        """## 6. Confusion matrix at the chosen operating point

We focus on the random forest (the most accurate model) at its F1-tuned
threshold.  The confusion matrix makes the precision/recall tradeoff
concrete in absolute counts.""",
    ),
    (
        "code",
        """from sklearn.metrics import confusion_matrix
chosen_name = 'RF (none)'
chosen_thr = tuned_thr[chosen_name]
y_pred = (test_proba[chosen_name] >= chosen_thr).astype(int)
cm = confusion_matrix(y_test, y_pred)
print(f'RF at F1-tuned threshold = {chosen_thr:.3f}')

fig, ax = plt.subplots(figsize=(4.5, 3.8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['pred = 0', 'pred = 1'],
            yticklabels=['actual = 0', 'actual = 1'], ax=ax)
ax.set_title(f'RF confusion matrix (thr = {chosen_thr:.3f})')
fig.tight_layout()
fig.savefig(FIG_DIR / '07_rf_confusion.png', dpi=200, bbox_inches='tight')
plt.show()""",
    ),
    (
        "markdown",
        """## 7. Feature importance

Two complementary views:

* **Logistic regression standardized coefficients** -- positive coefficients
  raise the predicted probability of food insecurity; negative coefficients
  lower it.  Magnitudes are comparable because the input features are
  standardized inside the pipeline.
* **Random forest impurity-based feature importances** -- captures
  nonlinear interactions but cannot tell us the *direction* of effect.""",
    ),
    (
        "code",
        """# Recover the standardized LR coefficients in the original feature order.
lr_pipe = search_lr_none.best_estimator_
lr_clf  = lr_pipe.named_steps['clf']
preproc = lr_pipe.named_steps['preproc']
feat_order_num = preproc.transformers_[0][2]
feat_order_bin = preproc.transformers_[1][2]
all_features = list(feat_order_num) + list(feat_order_bin)
coef_df = pd.DataFrame({'feature': all_features,
                        'coef': lr_clf.coef_.ravel()})
coef_df['abs'] = coef_df['coef'].abs()
top_lr = coef_df.sort_values('abs', ascending=False).head(20)

# Random forest importances.
rf_pipe = search_rf_none.best_estimator_
rf_clf  = rf_pipe.named_steps['clf']
rf_feats = list(rf_pipe.named_steps['preproc'].transformers_[0][2]) + \
           list(rf_pipe.named_steps['preproc'].transformers_[1][2])
rf_imp = pd.DataFrame({'feature': rf_feats,
                       'importance': rf_clf.feature_importances_})
top_rf = rf_imp.sort_values('importance', ascending=False).head(20)

fig, axes = plt.subplots(1, 2, figsize=(13, 7))
sns.barplot(data=top_lr.iloc[::-1], y='feature', x='coef',
            ax=axes[0], palette='RdBu_r', hue='feature', legend=False, dodge=False)
axes[0].axvline(0, color='black', linewidth=0.7)
axes[0].set_title('Logistic Regression: standardized coefficients (top 20 by |coef|)')
axes[0].set_xlabel('Coefficient')
axes[0].set_ylabel('')

sns.barplot(data=top_rf.iloc[::-1], y='feature', x='importance',
            ax=axes[1], palette='Greens_r', hue='feature', legend=False, dodge=False)
axes[1].set_title('Random Forest: impurity-based importances (top 20)')
axes[1].set_xlabel('Importance')
axes[1].set_ylabel('')
fig.tight_layout()
fig.savefig(FIG_DIR / '08_feature_importance.png', dpi=200, bbox_inches='tight')
plt.show()""",
    ),
    (
        "code",
        """# Save the top-feature tables for the report.
top_lr.to_csv(PROJECT_ROOT / 'artifacts' / 'lr_top_coefficients.csv', index=False)
top_rf.to_csv(PROJECT_ROOT / 'artifacts' / 'rf_top_importances.csv', index=False)
print('saved feature tables')""",
    ),
    (
        "markdown",
        """## 8. Operating-point summary

Two reference operating points to anchor the report's targeting
discussion:

* **Top-decile triage**: rank households by predicted risk and flag the
  top 10% as priority for outreach.  We report what fraction of true
  food-insecure households this captures (recall) and what share of the
  flagged households are truly food-insecure (precision).
* **Top-quintile triage**: same idea at top 20%.""",
    ),
    (
        "code",
        """def topk_metrics(y_true, y_proba, k_frac):
    n = len(y_proba)
    threshold = float(np.quantile(y_proba, 1 - k_frac))
    flagged = (y_proba >= threshold).astype(int)
    tp = int(((flagged == 1) & (y_true == 1)).sum())
    fp = int(((flagged == 1) & (y_true == 0)).sum())
    fn = int(((flagged == 0) & (y_true == 1)).sum())
    flagged_n = int(flagged.sum())
    precision = tp / max(1, flagged_n)
    recall = tp / max(1, int(y_true.sum()))
    return {
        'k_frac': k_frac,
        'threshold': round(threshold, 3),
        'flagged_n': flagged_n,
        'flagged_share': round(flagged_n / n, 3),
        'precision': round(precision, 3),
        'recall': round(recall, 3),
    }

op_rows = []
for base in ['LR (none)', 'SVM-Lin (none)', 'SVM-RBF (none)', 'RF (none)']:
    for k in [0.10, 0.20, 0.30]:
        row = {'model': base, **topk_metrics(y_test.values, test_proba[base], k)}
        op_rows.append(row)
op_df = pd.DataFrame(op_rows)
op_df.to_csv(PROJECT_ROOT / 'artifacts' / 'top_k_operating_points.csv', index=False)
op_df""",
    ),
    (
        "markdown",
        """## Outputs

* `artifacts/prediction_results_12row.csv` -- 12-row metrics table
  (4 models x 3 strategies);  also mirrored as `prediction_results_9row.csv`
* `artifacts/lr_top_coefficients.csv`, `artifacts/rf_top_importances.csv`
* `artifacts/top_k_operating_points.csv`
* `report/figures/05_pr_curves.png`
* `report/figures/06_calibration.png`
* `report/figures/07_rf_confusion.png`
* `report/figures/08_feature_importance.png`""",
    ),
]


if __name__ == "__main__":
    write_notebook("notebooks/02_prediction_track.ipynb", CELLS)
    print("wrote notebooks/02_prediction_track.ipynb")
