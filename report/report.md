# Predictive Targeting and Segmentation of Food-Insecure Households in the U.S. FoodAPS Survey

**EE 5290 - Final Project Report**
Iowa State University, Spring 2026
*May 14, 2026*

---

## Abstract

Food-assistance organizations face increasing demand under constrained
funding and capacity. Using a cleaned, household-level extract of the
USDA FoodAPS survey (4,826 households, 38 variables), we develop a
two-track data-analytics framework that helps such organizations
**(i)** predict which households are food insecure and
**(ii)** discover *types* of food-insecure households whose profiles
suggest distinct outreach strategies. We compare L2-regularized
logistic regression, an RBF-kernel SVM, and a random-forest classifier
under three class-imbalance strategies, and we use PCA followed by
K-means clustering, both with and without geographic features, to
expose household personas. On the held-out 20% test set, logistic
regression achieves PR-AUC 0.568 and ROC-AUC 0.783, with a top-decile
precision of 67% capturing 24% of food-insecure households.
Geography-free persona clustering identifies five interpretable
groups; 88% of the model's top-decile flags fall into two of them
(*non-working low-income singles* and *large low-income families with
children*), suggesting that targeted outreach can focus effort on
programs designed for these distinct populations.

---

## 1. Introduction

Food security - reliable access to nutritious food - is a foundational
component of household well-being. Across the Midwest in particular,
demand for assistance is rising while organisational capacity is not,
forcing food banks and outreach groups to make explicit choices about
*which* households to prioritize. This project asks two related
questions on the cleaned FoodAPS-derived dataset provided for the
case:

1. **Prediction.** Given household characteristics, can we predict
   whether the household is food insecure with enough accuracy to be
   useful for triage?
2. **Segmentation.** Among food-insecure households, what *kinds*
   of households exist, and how should that affect outreach?

We address both questions with library-first scikit-learn modelling,
focusing depth on (i) a controlled comparison of three classifiers
under three class-imbalance strategies, (ii) a careful choice of K
for K-means based on silhouette and stability, and (iii) a synthesis
section that joins the model's risk score to each household's persona
to produce concrete targeting recommendations.

**Contributions.**
1. A reusable preprocessing pipeline for the FoodAPS extract that
   recodes survey sentinel codes, encodes mixed nominal/ordinal
   features, and adds log-transformed access variables.
2. A head-to-head evaluation of logistic regression, RBF-kernel SVM
   and random forest on imbalanced food-insecurity prediction with
   operating-point analyses suited to a triage workflow.
3. Two complementary clusterings of the food-insecure subset: a
   geography-driven view and a five-persona, geography-free view.
4. A synthesis showing how the persona mix changes across the
   model's top-decile to top-half operating points.

**Non-causal framing.** The dataset is observational and a survey
sample. All findings are reported as *associations* between features
and the food-security outcome, never as causal effects.

---

## 2. Background and Methods

We summarize the four learning methods used in the report; full
derivations are standard in the course textbooks.

**Logistic regression.** Given features `x` in R^d and a binary outcome
`y` in {0,1}, the model `Pr(y=1|x) = sigma(w^T x + b)` with
`sigma(z) = 1/(1+exp(-z))` is fit by minimizing the negative
log-likelihood with L2 regularization controlled by `C`. The
coefficients `w` are interpretable on standardized inputs: `w_j > 0`
raises the predicted probability when feature `j` increases.

**Kernel SVM.** The dual form of the support vector machine
replaces inner products `x_i^T x_j` with a kernel `k(x_i, x_j)`. We
use the radial basis function (RBF) kernel
`k(x_i, x_j) = exp(-gamma ||x_i - x_j||^2)` with `gamma` controlling
locality and `C` the soft-margin penalty. Probabilities are produced
by Platt scaling on cross-validated margins.

**Random forest.** Ensembles of decision trees fit on bootstrap
samples with feature-bagging give a strong nonlinear baseline that
captures interactions automatically and yields impurity-based
feature importances.

**PCA and K-means.** PCA computes the top `k` singular vectors of
the centered standardized data matrix; the projection retains the
directions of maximum variance. K-means partitions samples into `k`
clusters by alternating assignment to the nearest centroid and
recomputation of centroids; we choose `k` by inspecting the elbow of
the within-cluster sum-of-squares (inertia) jointly with the
silhouette score.

**Class imbalance.** With a positive-class rate near 28%, accuracy is
misleading: a trivial all-negative classifier achieves 72% accuracy.
We therefore optimize and report *average precision* (PR-AUC) and
supplement it with *recall at fixed precision* thresholds. We
compare three strategies: no adjustment, balanced class weights
(cost-sensitive loss), and probability-threshold tuning chosen on
out-of-fold training predictions.

---

## 3. Data and Preprocessing

The provided dataset (`data/Case_dataset.csv`) contains 4,826
households described by 38 columns covering food-security outcomes,
economic conditions, household composition, employment and demographics,
food-assistance program participation, transportation and access
indicators, and food-acquisition behavior. The binary outcome is
`food_insecure_flag_adult`; 27.85% of the rows are positive
(see `figures/01_class_balance.png`). The four-level `adltfscat` and
the household-reported `foodsufficient` are excluded from the feature
matrix to avoid label leakage.

### 3.1 Cleaning decisions

Survey sentinel codes (-996, -997, -998) appear in `foodpantry`,
`anyvehicle`, `vehiclenum`, and `caraccess`; we recode them to NaN
before any analysis. Two columns are dropped entirely: `caraccess`
(92% sentinel-coded) and `fah_storetype_unique` (mean ~0.002,
near-constant). String-coded categorical heads (`head_hispanic`,
`head_racecat`) carry the response code "R" for "refused"; we
recode "R" to NaN. The survey weight `hhwgt` is documented but not
used in modelling (see Section 7).

### 3.2 Feature engineering

Distance variables (`dist_sm`, `dist_walmart`, `nearsnap_dist` and
similar) are right-skewed; we *add* a log(1+x) transform alongside
the raw value, so that linear models see a near-symmetric signal
while tree models retain the original units. Four household-composition
derivations are added: children per adult, elderly share, child share,
and employed-adult share. Nominal categoricals (`head_sex`,
`head_hispanic`, `head_racecat`, `head_employment`, `region`,
`poverty_band`, `targetgroup`) are one-hot encoded with NaN mapped
to an explicit "unknown" level. Numeric NaN are median-imputed. The
resulting matrix has 63 features after encoding.

### 3.3 Train/test split and EDA highlights

We hold out a stratified 20% of households as a final test set
(966 rows) using `random_state=42`. All hyperparameter tuning uses
5-fold stratified CV on the training fold only; the test set is
touched exactly once at the end of Section 4. See
`figures/01_class_balance.png` and `figures/02_marginal_rates.png`
for class balance and marginal food-insecurity rates by selected
features. Households at or below the poverty guideline have nearly a
50% positive rate; the rate is also visibly elevated for SNAP-using
households and households without an employed adult. Bars are
descriptive only and do not control for confounding.

---

## 4. Track 1: Predictive Modelling

We compare three classifiers under three imbalance strategies for a
total of nine evaluated configurations. For each base classifier we
run `GridSearchCV` over a small grid (LR: `C` in {0.01,0.1,1,10};
SVM-RBF: `C` in {0.5,1,5}, `gamma` in {scale,0.01,0.1}; RF:
`max_depth` in {None,10,20}, `min_samples_leaf` in {1,5} with 300
trees) using 5-fold stratified CV on the training fold and the
average-precision (PR-AUC) score function.

### 4.1 Threshold-tuning protocol

For each model, we use scikit-learn's `cross_val_predict` to generate
*out-of-fold* probability estimates on the training rows. We then
choose the probability threshold that maximizes F1 on those out-of-fold
predictions. This avoids using the test set to choose the threshold,
which would inflate the reported metrics.

### 4.2 Test-set results

| Model (strategy)         | thr   | ROC-AUC | PR-AUC | Brier | Prec. | Rec.  | F1    | R@P=.50 |
|--------------------------|-------|---------|--------|-------|-------|-------|-------|---------|
| LR - no adjust           | 0.50  | 0.7831  | 0.5681 | 0.161 | 0.624 | 0.364 | 0.460 | 0.610   |
| LR - balanced            | 0.50  | 0.7829  | 0.5681 | 0.195 | 0.463 | 0.762 | 0.576 | 0.598   |
| **LR - F1-tuned thr**    | 0.289 | 0.7831  | 0.5681 | 0.161 | 0.466 | 0.755 | **0.576** | 0.610 |
| SVM-RBF - no adjust      | 0.50  | 0.7551  | 0.5290 | 0.171 | 0.549 | 0.208 | 0.302 | 0.565   |
| SVM-RBF - balanced       | 0.50  | 0.7677  | 0.5269 | 0.166 | 0.564 | 0.346 | 0.429 | 0.651   |
| **SVM-RBF - F1-tuned**   | 0.236 | 0.7551  | 0.5290 | 0.171 | 0.425 | 0.747 | **0.542** | 0.565 |
| RF - no adjust           | 0.50  | 0.7667  | 0.5208 | 0.166 | 0.602 | 0.186 | 0.284 | 0.524   |
| RF - balanced            | 0.50  | 0.7665  | 0.5101 | 0.179 | 0.485 | 0.665 | 0.561 | 0.658   |
| **RF - F1-tuned thr**    | 0.301 | 0.7667  | 0.5208 | 0.166 | 0.453 | 0.792 | **0.577** | 0.524 |

Three observations stand out:

1. **Logistic regression wins on every aggregate metric**: ROC-AUC
   0.7831 and PR-AUC 0.5681. RF and SVM-RBF reach PR-AUC ~0.52. This
   is consistent with a problem whose signal is largely linear in the
   engineered features (a coincidence we exploit in 4.4).
2. **Class weights and threshold tuning are nearly equivalent and
   substantially better than the default 0.5 threshold** for the
   F1/recall combination. Both strategies trade roughly 17% of
   precision for an extra 30-50 percentage points of recall. The
   operationally relevant message is that the choice of threshold
   matters more than the choice of class-weight scheme.
3. **Recall at high precision is small for every method**: at a 70%
   precision floor, the best model recovers only ~26% of the
   food-insecure population. This sets a realistic expectation for
   the targeting recommendation in Section 6.

### 4.3 Visual diagnostics

`figures/05_pr_curves.png` shows the precision/recall curves for the
three classifiers' raw probability outputs. LR's curve dominates SVM
and RF in the high-recall regime, the regime that matters for a food
bank that wants to maximize coverage of the food-insecure population
subject to a precision floor. `figures/06_calibration.png` shows the
calibration diagnostics: LR is well calibrated across the range; the
random forest is mildly over-confident at the extremes; the SVM is
well-behaved in the middle but slightly under-confident at the low
end (an artefact of Platt scaling on a limited sample).

### 4.4 Interpretation

See `figures/08_feature_importance.png`. The two interpretation views
are remarkably consistent. Food-pantry use, low poverty band, and
the FoodAPS sampling target group dominate both rankings; the random
forest additionally elevates the cluster of distance variables,
reflecting that geographical access is the largest source of nonlinear
variation even after controlling for income. Logistic regression -
which can report *direction* - shows expected protective effects of
more vehicles, higher education, and elderly household members (the
latter likely reflects greater enrollment in income-stabilising
programs such as Social Security and SNAP). We use `foodpantry` as a
predictor because the data dictionary lists it under "program
participation"; we acknowledge it is a behavioural correlate of
insecurity rather than an independent causal driver.

---

## 5. Track 2: Segmentation of Food-Insecure Households

The 1,344 food-insecure households are projected onto principal
components of the standardized 63-feature matrix; 24 components
explain 80% of the variance. We then sweep K-means for `k` in {2..8}
with 20 random restarts per `k`.

### 5.1 Geographic clustering: a strong rural/urban axis

The silhouette criterion identifies `k=2` (silhouette = 0.221) as
the clear winner; `k=3..8` all hover near 0.10. The two clusters
correspond cleanly to rural (n=222, mean `dist_walmart` ~ 11 miles)
and non-rural (n=1,122, mean `dist_walmart` ~ 3 miles) households.
Stability across eight random initializations is high (mean ARI > 0.9).
This is a real but uninformative finding for outreach planning: it
tells the food bank that physical access dominates within-population
variation but not what programmes to design within each access tier.

### 5.2 Geography-free persona clustering

We therefore re-cluster after dropping the seven geographic features
(distances, rural, nonmetro, region one-hots). The remaining 50
features capture household composition, employment, demographics,
poverty, program use, and food-acquisition behaviour. Silhouette is
maximized at `k=2` as before but more usable values appear at
`k` in {3,4,5}; we cap at `k=5` for narrative tractability and
report that solution. See `figures/14_persona_heatmap.png`.

The five personas admit clean narrative names:

- **P0 - Working families just above poverty** (n=266, 19.8% of the
  food-insecure subset). Nearly all have an employed adult; poverty
  ratio 1.43; moderate household size; pantry use only 9%.
- **P1 - Non-working low-income singles, high pantry use** (n=391,
  29.1%). Small households (mean size 2.0); only 16% have an employed
  adult; poverty ratio 0.85; pantry use 28% - triple the food-insecure
  average. This is the cluster most clearly served by direct food
  distribution.
- **P2 - Large low-income families with children** (n=347, 25.8%).
  Mean household size 5.0 with 2.6 children; poverty ratio 0.85;
  partial employment; lower head education. School-meal eligibility
  is highest in this cluster.
- **P3 - Working educated households near the boundary** (n=214,
  15.9%). Highest poverty ratio (3.05, well *above* the poverty
  guideline) and highest education; near-universal employment. These
  are households whose food insecurity is unlikely to be resolved by
  income-eligibility programmes.
- **P4 - Elderly low-income households, high SNAP enrollment** (n=126,
  9.4%). Mean head age 70; lowest employment; highest SNAP enrolment
  (42%). Their inclusion in the food-insecure subset suggests SNAP
  is necessary but not always sufficient.

---

## 6. Synthesis: Targeting Recommendations

We assign every household in the test set to a persona by passing it
through the same preprocessing -> behaviour-only PCA -> K-means
pipeline used in Section 5.2, and rank households by the LR model's
predicted probability. This lets us look at *who* the model would
actually be flagging at each operating point.

See `figures/15_persona_mix_at_cutoffs.png` and
`figures/16_precision_recall_vs_cutoff.png`.

**Top-decile composition.** At the top 10% predicted-risk cutoff,
precision is 68% (97 of the 966 test households are flagged; 66 are
truly food-insecure, capturing 24% of the test set's food-insecure
population). Of those 97 flagged households, **45% are P1 (non-working
low-income singles) and 43% are P2 (large low-income families with
children)**. The two clusters whose risk the model rates lowest -
P3 (working educated near boundary) and P4 (elderly on SNAP) -
together account for under 2% of the top-decile flags despite
making up over half the population.

**Operational implication.** A food bank that adopts top-decile triage
of the model can plan outreach as if the workload would split evenly
between two distinct programme tracks: (a) supplemental food
distribution for low-income adult singles (potentially via existing
pantry partnerships, since this group already over-uses pantries);
and (b) family-oriented programmes for large low-income households
with children, including school-meal coordination and bulk-pack
distributions. The remaining 12% of the top decile sits in P0
(working families just above poverty) and warrants light-touch
SNAP-enrolment outreach.

**What the model misses.** Persona P3 (working educated households
just above the poverty line) is 36.9% of the population but only 15.9%
of the food-insecure subset; the model rates them with mean probability
0.12 and they account for <0.5% of the top-decile flags. These
households may experience episodic insecurity that the survey's
snapshot does not capture well, and they are unlikely to be reached
by income-eligibility programmes. This is a population that warrants
qualitative follow-up rather than algorithmic targeting.

**Equity inspection.** Persona P4 (elderly on SNAP) is under-represented
in the top decile relative to its share of food-insecure households
(1.0% versus 9.4%). Two interpretations are consistent with the data:
the model correctly de-prioritizes households whose insecurity is
partly already addressed by SNAP and Social Security; or the model
under-weights this group because the snapshot fails to capture
elderly-specific access barriers (mobility, dietary restrictions).
The recommendation is to monitor this group via existing SNAP
partnerships rather than treat the model's low scores as a green
light to disengage.

---

## 7. Limitations and Future Work

**Survey weights ignored.** The provided dataset is a probability
sample with weights in `hhwgt`. We did not use them in fitting or in
evaluation, on the basis of the data dictionary's note that weights
are optional. A weighted retrain - using `sample_weight` in the LR
loss and a weighted scoring function - would produce different
coefficient magnitudes and may shift cluster sizes; we recommend
this as the first follow-up.

**Observational data.** Every coefficient and importance is an
*association*, not a causal effect. In particular, the strong
positive coefficient on `foodpantry` is best read as "households
that already use pantries are also reporting food insecurity", not
"pantry use causes food insecurity". The same caveat applies to
`snap_any`: its protective sign here cannot be read as a causal
estimate of SNAP's effectiveness.

**Imbalance methods left unexplored.** We compared no adjustment,
balanced class weights, and threshold tuning, and found them roughly
equivalent in F1 with no method dominating across precision floors.
Future work should add SMOTE or its variants for synthetic minority
oversampling and cost-sensitive losses with explicit cost matrices,
and should evaluate them on a Bayesian decision-theoretic objective
that weights false-negative cost (missing a food-insecure household)
against false-positive cost (wasted outreach effort).

**Persona stability across outcome thresholds.** Our personas are
fit on households flagged on the binary outcome. Refitting on the
4-level `adltfscat` (low vs. very low food security) and on
near-poverty households who narrowly avoided the binary flag would
test whether the personas track *insecurity* or *poverty*.

**External data.** The case allows supplementing with USDA county-level
access data and ACS demographics. Adding even a single county-level
summary (e.g., median income, supermarket density) would plausibly
tighten the model and let us sanity-check distance features against
published food-desert maps.

**Reproducibility.** All results are reproduced by
`notebooks/01..04.ipynb` with `random_state=42` and `requirements.txt`
pinned to scikit-learn 1.8. See `README.md` for setup.

---

## 8. Conclusion

A library-first analysis of the FoodAPS extract is sufficient to
build both a useful predictive model (PR-AUC 0.57, top-decile
precision 67%) and an interpretable five-persona segmentation of
the food-insecure population. The synthesis of the two - reading off
*who* the model flags at the top decile - turns an abstract ranking
into concrete programme recommendations: roughly half the top decile
is non-working low-income singles, half is large low-income families
with children, and the remaining 12% is working families on the
poverty boundary. This kind of analysis is exactly the type of
decision support that capacity-constrained food banks need; the
methodology generalizes to other household-survey datasets with
mixed nominal/ordinal features and an imbalanced binary outcome.

---

## References

1. F. Pedregosa et al. "Scikit-learn: Machine Learning in Python",
   *JMLR*, vol. 12, pp. 2825-2830, 2011.
2. K. P. Murphy. *Probabilistic Machine Learning: An Introduction*.
   MIT Press, 2022.
3. S. Boyd and L. Vandenberghe. *Introduction to Applied Linear Algebra*.
   Cambridge University Press, 2018.
4. A. Blum, J. Hopcroft, and R. Kannan. *Foundations of Data Science*.
   Cambridge University Press, 2020.
5. USDA Economic Research Service. "National Household Food Acquisition
   and Purchase Survey (FoodAPS)". Accessed May 2026.
