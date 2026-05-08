"""Generate notebooks/03_segmentation_track.ipynb."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_notebook import write_notebook  # noqa: E402

CELLS = [
    (
        "markdown",
        """# 03 - Track 2: Segmentation of Food-Insecure Households

This notebook is the segmentation track of the EE 5290 final project.
Among the ~1,300 food-insecure households in the cleaned dataset, we want
to discover distinct *types* whose profiles can inform targeted outreach.

The pipeline is:

1. filter to ``food_insecure_flag_adult == 1``
2. standardize the same numeric and one-hot feature matrix used by the
   prediction track (minus the label)
3. fit PCA, retain enough components to explain ~80% of variance
4. sweep K-Means for `k` in 2..8, choose `k` via silhouette and inertia
5. assess cluster stability across multiple random initializations
6. profile each cluster on the original (un-standardized) features and
   give each cluster a narrative name""",
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
    OUTCOME, load_raw, clean, build_feature_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
""",
    ),
    (
        "markdown",
        """## 1. Build the food-insecure subset

We re-run the same `data_prep` pipeline so segmentation uses identical
features to the prediction track.  We then filter to food-insecure
households only -- ~1,344 rows.""",
    ),
    (
        "code",
        """raw = load_raw('data/Case_dataset.csv')
cleaned = clean(raw)
fm = build_feature_matrix(cleaned)

mask = (fm.y == 1)
X_fi = fm.X[mask].reset_index(drop=True)
print('food-insecure subset shape:', X_fi.shape)

# Keep an aligned slice of cleaned rows for un-standardized profiling.
profile_df = cleaned.loc[mask].reset_index(drop=True)
print('aligned profile rows:', profile_df.shape)""",
    ),
    (
        "code",
        """# Standardize for PCA / K-Means.  All columns of fm.X are numeric (binary or
# continuous).  We standardize all columns so that no single feature scale
# dominates the Euclidean distance.  Note that this puts binary columns on
# the same footing as continuous ones -- a common compromise when no
# domain reason exists to weight them differently.
scaler = StandardScaler()
X_fi_std = pd.DataFrame(
    scaler.fit_transform(X_fi),
    columns=X_fi.columns,
)
print('standardized matrix shape:', X_fi_std.shape)""",
    ),
    (
        "markdown",
        """## 2. PCA: how many components do we need?

We fit PCA up to the rank of the matrix and look at cumulative explained
variance.  We pick the smallest dimensionality that explains at least
80% of the variance for the K-Means input space.""",
    ),
    (
        "code",
        """pca_full = PCA(random_state=RANDOM_STATE).fit(X_fi_std)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components_80 = int(np.searchsorted(cum_var, 0.80) + 1)
print(f'components needed to reach 80% var: {n_components_80}')

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(np.arange(1, len(cum_var) + 1), cum_var, marker='o', markersize=3)
ax.axhline(0.80, linestyle='--', color='gray', linewidth=1, label='80% target')
ax.axvline(n_components_80, linestyle='--', color='red', linewidth=1,
           label=f'k = {n_components_80}')
ax.set_xlabel('Number of principal components')
ax.set_ylabel('Cumulative explained variance')
ax.set_title('PCA on the food-insecure subset')
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / '09_pca_variance.png', dpi=200, bbox_inches='tight')
plt.show()""",
    ),
    (
        "code",
        """# Inspect the loadings of the top 3 PCs to see what they encode.
pca = PCA(n_components=n_components_80, random_state=RANDOM_STATE).fit(X_fi_std)
X_pca = pca.transform(X_fi_std)
print('PCA-reduced shape:', X_pca.shape)

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(pca.n_components_)],
    index=X_fi.columns,
)

def top_loadings(loadings, pc, k=8):
    s = loadings[pc].abs().sort_values(ascending=False).head(k)
    return loadings.loc[s.index, [pc]].sort_values(pc)

for pc in ['PC1', 'PC2', 'PC3']:
    print(f'--- {pc} top |loading| features ---')
    print(top_loadings(loadings, pc).round(3).to_string())
    print()""",
    ),
    (
        "markdown",
        """## 3. K-Means sweep: choose `k` by silhouette + inertia

We fit K-Means in the PCA-reduced space for `k` in 2..8 with 20
restarts each (`n_init=20`).  We report inertia (within-cluster
distance, smaller is tighter clusters) and silhouette score (higher is
better separation, range -1..1).""",
    ),
    (
        "code",
        """k_values = list(range(2, 9))
sweep_rows = []
fitted = {}
for k in k_values:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit(X_pca)
    sil = silhouette_score(X_pca, km.labels_, sample_size=min(2000, X_pca.shape[0]),
                            random_state=RANDOM_STATE)
    sweep_rows.append({'k': k, 'inertia': km.inertia_, 'silhouette': sil})
    fitted[k] = km

sweep_df = pd.DataFrame(sweep_rows)
sweep_df""",
    ),
    (
        "code",
        """fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(sweep_df['k'], sweep_df['inertia'], marker='o')
axes[0].set_title('Inertia (elbow plot)')
axes[0].set_xlabel('k')
axes[0].set_ylabel('Within-cluster sum of squares')

axes[1].plot(sweep_df['k'], sweep_df['silhouette'], marker='o', color='#C44E52')
axes[1].set_title('Silhouette score')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Silhouette (higher is better)')

best_k = int(sweep_df.loc[sweep_df['silhouette'].idxmax(), 'k'])
axes[1].axvline(best_k, linestyle='--', color='gray', linewidth=1,
                label=f'best k = {best_k}')
axes[1].legend()
fig.suptitle('K-Means model selection (PCA-reduced space)', y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / '10_kmeans_sweep.png', dpi=200, bbox_inches='tight')
plt.show()
print('best silhouette at k =', best_k)""",
    ),
    (
        "markdown",
        """## 4. Stability check across random seeds

A single K-Means fit can be sensitive to initialization.  We refit at
the chosen `k` with 8 different random states and compute pairwise
adjusted Rand index (ARI) between the resulting label assignments.
ARI = 1.0 means identical clusterings up to permutation; ARI close to
0 means the cluster structure is unstable.""",
    ),
    (
        "code",
        """seeds = [0, 7, 13, 21, 42, 84, 99, 123]
labelings = []
for s in seeds:
    km = KMeans(n_clusters=best_k, random_state=s, n_init=20).fit(X_pca)
    labelings.append(km.labels_)

ari = np.zeros((len(seeds), len(seeds)))
for i in range(len(seeds)):
    for j in range(len(seeds)):
        ari[i, j] = adjusted_rand_score(labelings[i], labelings[j])

fig, ax = plt.subplots(figsize=(5.0, 4.2))
sns.heatmap(ari, ax=ax, vmin=0.5, vmax=1.0, cmap='YlGnBu', annot=True, fmt='.2f',
            xticklabels=seeds, yticklabels=seeds, cbar_kws={'label': 'ARI'})
ax.set_title(f'K-Means stability across seeds (k = {best_k})')
fig.tight_layout()
fig.savefig(FIG_DIR / '11_kmeans_stability.png', dpi=200, bbox_inches='tight')
plt.show()

mean_ari = ari[np.triu_indices_from(ari, k=1)].mean()
print(f'mean off-diagonal ARI: {mean_ari:.3f}')""",
    ),
    (
        "markdown",
        """## 5. Cluster profiles on un-standardized features

We assign each food-insecure household to a cluster using the K-Means
fit at the chosen `k` and `random_state=RANDOM_STATE`, then compute
mean / proportion of selected interpretable features per cluster.""",
    ),
    (
        "code",
        """km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20).fit(X_pca)
profile_df = profile_df.copy()
profile_df['cluster'] = km_final.labels_

# Profile features chosen for narrative interpretability.
profile_features = [
    'household_size', 'num_adults', 'num_children', 'num_elderly',
    'employed_adults', 'any_employed_adult',
    'head_age', 'head_educcat',
    'poverty_ratio', 'snap_any', 'foodpantry',
    'rural', 'nonmetro',
    'anyvehicle', 'vehiclenum',
    'dist_sm', 'dist_walmart', 'nearsnap_dist',
    'fah_event_count', 'fafh_event_count',
]
profile_features = [c for c in profile_features if c in profile_df.columns]

profile = profile_df.groupby('cluster')[profile_features].mean().round(2)
profile['n'] = profile_df.groupby('cluster').size()
profile""",
    ),
    (
        "code",
        """# Add the share of each cluster within the food-insecure subset.
profile['share'] = (profile['n'] / profile['n'].sum()).round(3)
# And cross-tab against region / poverty band for narrative naming.
region_xt = pd.crosstab(profile_df['cluster'], profile_df['region'], normalize='index').round(2)
region_xt.columns = [f'region={c}' for c in region_xt.columns]
poverty_xt = pd.crosstab(profile_df['cluster'], profile_df['poverty_band'], normalize='index').round(2)

cluster_table = pd.concat([profile, region_xt], axis=1)
cluster_table""",
    ),
    (
        "code",
        """print('--- poverty-band shares within each cluster ---')
print(poverty_xt)""",
    ),
    (
        "code",
        """# Save full cluster profile for the report.
cluster_table.to_csv(PROJECT_ROOT / 'artifacts' / 'cluster_profile.csv')
poverty_xt.to_csv(PROJECT_ROOT / 'artifacts' / 'cluster_poverty_xt.csv')
print('saved cluster_profile.csv and cluster_poverty_xt.csv')""",
    ),
    (
        "markdown",
        """## 6. Visual cluster profiles

A heatmap of the *standardized* mean of each profile feature within each
cluster makes it easy to read which clusters are above or below the
food-insecure-subset average for each feature.  Red cells mean
"distinctively higher than the food-insecure average"; blue cells mean
"lower than the food-insecure average".""",
    ),
    (
        "code",
        """profile_means = profile_df.groupby('cluster')[profile_features].mean()
overall_means = profile_df[profile_features].mean()
overall_stds = profile_df[profile_features].std().replace(0, 1.0)
profile_z = (profile_means - overall_means) / overall_stds

fig, ax = plt.subplots(figsize=(11, 4.5))
sns.heatmap(profile_z.T, ax=ax, cmap='RdBu_r', center=0, annot=True, fmt='.1f',
            cbar_kws={'label': 'z-score vs. food-insecure mean'},
            xticklabels=[f'C{c} (n={profile.loc[c, "n"]:.0f}, '
                          f'{profile.loc[c, "share"]:.0%})' for c in profile.index])
ax.set_title('Cluster profiles among food-insecure households (z-scored)')
ax.set_xlabel('Cluster')
ax.set_ylabel('Feature')
fig.tight_layout()
fig.savefig(FIG_DIR / '12_cluster_heatmap.png', dpi=200, bbox_inches='tight')
plt.show()""",
    ),
    (
        "markdown",
        """## 7. Cluster scatter in PCA space

A 2-D scatter of the food-insecure subset projected onto PC1/PC2,
coloured by assigned cluster.  Useful for the report's segmentation
figure even though the actual clustering is performed in higher-dim
PCA space.""",
    ),
    (
        "code",
        """fig, ax = plt.subplots(figsize=(6.0, 4.8))
palette = sns.color_palette('Set2', n_colors=best_k)
for c in range(best_k):
    pts = X_pca[km_final.labels_ == c]
    ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.5,
                color=palette[c], label=f'C{c} (n={(km_final.labels_==c).sum()})')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)')
ax.set_title(f'K-Means clusters of food-insecure households (k = {best_k})')
ax.legend(fontsize=8, markerscale=1.6, loc='best')
fig.tight_layout()
fig.savefig(FIG_DIR / '13_cluster_scatter.png', dpi=200, bbox_inches='tight')
plt.show()""",
    ),
    (
        "markdown",
        """## 8. Narrative cluster names

We assign each cluster a short narrative name based on its z-scored
profile.  These names are produced semi-automatically by the rules
below; the final report describes how to interpret them.""",
    ),
    (
        "code",
        """def name_cluster(z_row, profile_row):
    fragments = []
    if z_row.get('rural', 0) > 0.5 or z_row.get('nonmetro', 0) > 0.5:
        fragments.append('Rural')
    elif z_row.get('rural', 0) < -0.5:
        fragments.append('Urban')
    if z_row.get('num_elderly', 0) > 0.5 and z_row.get('head_age', 0) > 0.3:
        fragments.append('elderly-headed')
    elif z_row.get('num_children', 0) > 0.5:
        fragments.append('with children')
    elif z_row.get('any_employed_adult', 0) > 0.3:
        fragments.append('working-age')
    if profile_row.get('snap_any', 0) > 0.6:
        fragments.append('on SNAP')
    elif z_row.get('snap_any', 0) < -0.5:
        fragments.append('not on SNAP')
    if z_row.get('vehiclenum', 0) < -0.5 or z_row.get('anyvehicle', 0) < -0.5:
        fragments.append('limited vehicle access')
    if not fragments:
        fragments.append('mixed profile')
    return ' '.join(fragments).capitalize()

names = {}
for c in profile.index:
    names[c] = name_cluster(profile_z.loc[c], profile.loc[c])

names_df = pd.DataFrame({
    'cluster': list(names.keys()),
    'narrative_name': list(names.values()),
    'n': [profile.loc[c, 'n'] for c in names],
    'share': [profile.loc[c, 'share'] for c in names],
})
names_df.to_csv(PROJECT_ROOT / 'artifacts' / 'cluster_names.csv', index=False)
names_df""",
    ),
    (
        "markdown",
        """## 9. Geography-free clustering for household personas

The clustering above is dominated by the rural/urban + distance axis,
which absorbs almost all of the silhouette signal at `k = 2`.  This is
informative -- *geographic access* is the single biggest source of
variation among food-insecure households -- but for outreach planning
we also want a *within-geography* persona view.

To expose finer household-composition and program-participation
segments, we re-cluster after dropping all geography-related features
(distances, rural/nonmetro, region).  The remaining feature space
captures household composition, employment, head demographics, poverty,
program use, and food-acquisition behavior.""",
    ),
    (
        "code",
        """geo_substrings = ['dist_', 'nearsnap', 'nearff', 'nearnonff',
                  'rural', 'nonmetro', 'region=']
behavior_columns = [c for c in X_fi.columns
                    if not any(s in c for s in geo_substrings)]
print(f'kept {len(behavior_columns)} of {X_fi.shape[1]} columns for persona clustering')

X_beh_std = StandardScaler().fit_transform(X_fi[behavior_columns])
pca_beh = PCA(random_state=RANDOM_STATE).fit(X_beh_std)
n_beh = int(np.searchsorted(np.cumsum(pca_beh.explained_variance_ratio_), 0.80) + 1)
print(f'persona PCA: {n_beh} components for 80% variance')
X_beh_pca = PCA(n_components=n_beh, random_state=RANDOM_STATE).fit_transform(X_beh_std)

beh_rows = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20).fit(X_beh_pca)
    sil = silhouette_score(X_beh_pca, km.labels_,
                            sample_size=min(2000, X_beh_pca.shape[0]),
                            random_state=RANDOM_STATE)
    beh_rows.append({'k': k, 'inertia': km.inertia_, 'silhouette': sil})
beh_sweep = pd.DataFrame(beh_rows)
beh_sweep""",
    ),
    (
        "code",
        """# Choose the k that maximizes silhouette for personas.  Cap at k=5 for
# narrative tractability (the report can describe at most ~5 clusters in
# the available page budget).
beh_best_k = int(beh_sweep[beh_sweep['k'] <= 5]
                  .loc[beh_sweep['silhouette'].idxmax(), 'k'])
print('persona best k (capped at 5):', beh_best_k)

km_beh = KMeans(n_clusters=beh_best_k, random_state=RANDOM_STATE, n_init=20).fit(X_beh_pca)
profile_df['persona'] = km_beh.labels_

persona_features = [
    'household_size', 'num_adults', 'num_children', 'num_elderly',
    'employed_adults', 'any_employed_adult',
    'head_age', 'head_educcat',
    'poverty_ratio', 'snap_any', 'foodpantry',
    'fah_event_count', 'fafh_event_count', 'fafh_schoolmeal_events',
]
persona_features = [c for c in persona_features if c in profile_df.columns]
persona_profile = profile_df.groupby('persona')[persona_features].mean().round(2)
persona_profile['n'] = profile_df.groupby('persona').size()
persona_profile['share'] = (persona_profile['n'] / persona_profile['n'].sum()).round(3)
persona_profile""",
    ),
    (
        "code",
        """# z-scored persona heatmap (within food-insecure subset).
persona_means = profile_df.groupby('persona')[persona_features].mean()
overall_p = profile_df[persona_features].mean()
overall_s = profile_df[persona_features].std().replace(0, 1.0)
persona_z = (persona_means - overall_p) / overall_s

fig, ax = plt.subplots(figsize=(11, 4.0))
sns.heatmap(persona_z.T, ax=ax, cmap='RdBu_r', center=0, annot=True, fmt='.1f',
            cbar_kws={'label': 'z-score vs. food-insecure mean'},
            xticklabels=[f'P{c} (n={persona_profile.loc[c, "n"]:.0f}, '
                          f'{persona_profile.loc[c, "share"]:.0%})'
                          for c in persona_profile.index])
ax.set_title('Geography-free persona clusters (z-scored)')
ax.set_xlabel('Persona')
ax.set_ylabel('Feature')
fig.tight_layout()
fig.savefig(FIG_DIR / '14_persona_heatmap.png', dpi=200, bbox_inches='tight')
plt.show()""",
    ),
    (
        "code",
        """# Cross-tab personas against the geography clusters and the rural flag, to
# show how the two clusterings interact.
xt_geo = pd.crosstab(profile_df['persona'], profile_df['cluster'], normalize='index').round(2)
xt_geo.columns = [f'geo-cluster {c}' for c in xt_geo.columns]
xt_rural = profile_df.groupby('persona')['rural'].mean().round(2).rename('rural_share')
xt_snap = profile_df.groupby('persona')['snap_any'].mean().round(2).rename('snap_share')
xt_pantry = profile_df.groupby('persona')['foodpantry'].mean().round(2).rename('pantry_share')
xt_combined = pd.concat([persona_profile[['n', 'share']], xt_rural, xt_snap, xt_pantry, xt_geo], axis=1)
xt_combined.to_csv(PROJECT_ROOT / 'artifacts' / 'persona_profile.csv')
xt_combined""",
    ),
    (
        "markdown",
        """## Outputs

* `artifacts/cluster_profile.csv` -- geographic cluster x feature mean table
* `artifacts/cluster_poverty_xt.csv` -- cluster x poverty-band shares
* `artifacts/cluster_names.csv` -- narrative cluster names
* `artifacts/persona_profile.csv` -- geography-free persona profile
* `report/figures/09_pca_variance.png`
* `report/figures/10_kmeans_sweep.png`
* `report/figures/11_kmeans_stability.png`
* `report/figures/12_cluster_heatmap.png`
* `report/figures/13_cluster_scatter.png`
* `report/figures/14_persona_heatmap.png`""",
    ),
]


if __name__ == "__main__":
    write_notebook("notebooks/03_segmentation_track.ipynb", CELLS)
    print("wrote notebooks/03_segmentation_track.ipynb")
