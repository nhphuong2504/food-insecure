# Presentation Script -- EE 5290 Final Project

Companion to `report/slides.pptx` (14 slides, 16:9 widescreen).
Target run time: **12-15 minutes** of talk + 3-5 minutes Q&A.

The script applies the data-storytelling arc:
**Hook -> Context -> Discovery -> Climax -> Recommendation -> Ask.**

Italics are stage directions or transitions. **Bold** is what to land hard.
Time hints are rough.

---

## Slide 1 -- Title  *(~20 s)*

> Hi everyone. My project is on **predictive targeting and segmentation
> of food-insecure households** in the Midwest. The deliverable is one
> sentence long: take a real food-bank dataset, and tell the food bank
> *who* to help and *what* to fund.
>
> I'll get to the recommendation in about ten minutes.

*(Click forward.)*

---

## Slide 2 -- The Hook  *(~75 s)*  **[the headline insight, up front]**

> Three numbers. They carry the whole talk.
>
> **27.85%.** That's the share of households in the dataset flagged as
> food insecure. One in four. That's the magnitude of the problem.
>
> **67%.** That's the precision of my best model when it flags the top
> ten percent of households by risk score. Two in three of those flags
> are correct -- which is good enough to drive triage, and an order of
> magnitude better than random.
>
> **88%.** This is the one I want you to remember. When the model picks
> the top ten percent, **88% of those flagged households fall into just
> two of the five personas I'll describe.** That number is what turns
> "we have a list" into "we know what programs to fund."
>
> The rest of the talk explains those three numbers and what to do
> with them.

*(Transition.)* "Let's set the scene."

---

## Slide 3 -- The Problem  *(~45 s)*

> Real operational pressure: across the Midwest, **demand for food
> assistance is rising while organisational capacity is fixed.** Food
> banks are forced to choose -- which households, which programs.
>
> So I split the project into the two questions a food bank actually
> needs answered. **Prediction**: who is at risk? **Segmentation**:
> what kinds of at-risk households are there? Two questions, two tracks,
> joined by a synthesis at the end.

*(Click.)*

---

## Slide 4 -- The Data  *(~60 s)*

> The provided extract is **4,826 households**, 38 features. The
> outcome is `food_insecure_flag_adult`, a binary flag, **27.85%
> positive**. That right there tells us we cannot trust raw accuracy
> -- a model that always predicts zero gets 72%. **PR-AUC** is the
> metric we'll judge models by, throughout.
>
> Three cleaning details mattered. The data uses **survey sentinel
> codes** -- minus 996, minus 997, minus 998 -- meaning "refused" or
> "not in universe". I recoded those to NaN. One column, `caraccess`,
> was 92% sentinel; I dropped it. And then the standard moves --
> log-transformed distances, one-hot encoded nominals, median-imputed
> the rest, **stratified 80/20 split, all tuning on the train fold
> only.** The test set gets touched exactly once.

---

## Slide 5 -- What We Already Knew  *(~60 s)*

> Before any modelling, this slide is just descriptive. Marginal
> food-insecurity rates by selected features.
>
> The pattern is what you'd expect, but the magnitudes are the point.
> **Households at or below the poverty line are about 47% positive --
> nearly double the overall rate.** SNAP-using households, households
> without an employed adult -- all visibly elevated. Pantry use roughly
> triples the rate.
>
> Important caveat I'll only say once: these are **bivariate**
> associations. They tell us where to look, not what causes what.
> That's why we go to a model next.

*(Transition.)* "Two tracks. Methods on one slide."

---

## Slide 6 -- The Approach  *(~50 s)*

> Track 1 on the left -- prediction. Three classifiers crossed with
> three imbalance strategies, so a **9-cell results matrix**.
> Logistic regression, SVM with RBF, random forest. None, balanced
> weights, threshold-tuned. Tuning metric is PR-AUC. Threshold is
> chosen on training out-of-fold predictions only.
>
> Track 2 on the right -- segmentation. I filter to the 1,344
> food-insecure households, run PCA, sweep K-means k from two to
> eight, choose by silhouette and elbow. **I ran clustering twice** --
> once with geography included, once without. I'll explain why on
> slide ten.

---

## Slide 7 -- Track 1 Result  *(~75 s)*  **[first climax]**

> Three takeaways from the prediction track.
>
> **One: logistic regression wins.** PR-AUC of **0.568**, ROC-AUC of
> **0.78**. SVM and random forest land at 0.529 and 0.521 on PR-AUC
> -- a clear gap. That's a slightly surprising result; you'd expect
> the random forest to dominate. What it tells us is that **the
> signal in this dataset, after my feature engineering, is largely
> linear.** Which means I get to keep an interpretable model.
>
> **Two: threshold tuning roughly equals balanced weights.** Both
> push F1 from about 0.46 up to 0.58. The lesson is that **on this
> kind of imbalance, the threshold matters more than the weighting
> scheme.**
>
> **Three: there's a real ceiling.** If you demand 70% precision,
> the best model recovers only about a quarter of the food-insecure
> population. That ceiling shows up again in the recommendation.

---

## Slide 8 -- The Operating Point  *(~50 s)*

> Two diagnostic plots that justify why I pick LR.
>
> Left: **logistic regression dominates the high-recall regime that
> matters for triage.** That's the regime a food bank operates in --
> they want to find as many at-risk households as possible subject
> to a precision floor.
>
> Right: calibration. LR hugs the diagonal -- so when it says 0.7
> probability, the empirical positive rate at 0.7 actually is around
> 0.7. That matters: probability scores are usable, not just rankings.

---

## Slide 9 -- What Drives It  *(~60 s)*

> This is my favourite Track 1 slide, because both views agree.
>
> **Left**, signed logistic regression coefficients: poverty band,
> food-pantry use, and the FoodAPS sampling target raise predicted
> risk. Education and number of vehicles **lower** it. Number of
> elderly is mildly protective -- which I read as: elderly households
> are more reliably enrolled in income-stabilising programs like
> Social Security and SNAP.
>
> **Right**, random forest impurity importances: same top features,
> plus the cluster of distance variables shows up higher -- that's
> the **nonlinear** access component a linear model can't capture.
>
> The point: linear and tree-based interpretations agree. That's the
> strongest evidence that the signal is real and not a modelling
> artefact.

*(Transition.)* "OK -- onto the second track."

---

## Slide 10 -- Track 2 Discovery  *(~60 s)*

> The segmentation track answers: *what kinds* of food-insecure
> households exist?
>
> Standard pipeline -- standardize, PCA, K-means sweep. The chart on
> the right is the silhouette and elbow analysis.
>
> Here's the subtlety I promised earlier. When I included distance
> and rural/region features, **silhouette overwhelmingly preferred
> k=2** -- the data has one dominant axis, which is rural versus
> urban access. That's a real finding, but it's **not actionable**
> for outreach planning -- you can't run a "rural" program.
>
> So I rebuilt the clustering on just household composition,
> employment, demographics, poverty band, programs, and food-acquisition
> behaviour. Silhouette and elbow agree on **k=5**. Eight random
> initializations agree -- mean ARI above 0.9, so the personas are
> stable.

---

## Slide 11 -- The Personas  *(~75 s)*

> Five personas. I'll walk the table on the right.
>
> **P0**, working families just above poverty -- 20% of the food-insecure.
> Almost all employed; poverty ratio 1.43.
>
> **P1**, non-working low-income singles -- 29%, the largest persona.
> Small households, only 16% with an employed adult, **pantry use 28%
> -- triple the food-insecure average.** This is the cluster that
> already finds its way to the existing pantry network.
>
> **P2**, large low-income families with children -- 26%. Mean
> household size five, with 2.6 children. Poverty ratio 0.85. Lowest
> head education. School-meal eligibility highest here.
>
> **P3**, working educated households just above poverty -- 16%.
> Poverty ratio 3.05 -- they're *above* the poverty line. They're
> educated, employed, and still flagged food insecure on the survey.
> They likely experience **episodic** insecurity. Important: they're
> unlikely to qualify for income-eligibility programs.
>
> **P4**, elderly low-income on SNAP -- 9%. Mean head age 70, 42% on
> SNAP. They are food insecure *despite* SNAP, which is interesting
> in itself.

---

## Slide 12 -- The Climax  *(~75 s)*  **[the synthesis]**

> Now we synthesise. I take every household in the test set, assign
> it to a persona, and rank by the LR model's predicted probability.
>
> Look at the leftmost bar of the chart on the right -- the top 10%
> cutoff. **45% of the flagged households are P1, and 43% are P2.**
> Two personas, **88% of the flags.**
>
> *(Pause.)*
>
> That's the climax of the talk. The model isn't just producing a list.
> It's pointing at two specific operational populations. **You don't
> need five programs. You need two -- and the model tells you exactly
> who they are.**

---

## Slide 13 -- The Recommendation  *(~75 s)*

> So here's what you'd tell a real Midwest food bank.
>
> **Program A**, in orange: pantry partnerships for non-working
> low-income singles. P1, 45% of top-decile flags. These households
> already self-select into pantries -- so meet them where they are.
> Lift is operational, not net-new infrastructure.
>
> **Program B**, in navy: family bulk packs and school-meal
> coordination for large low-income families with children. P2, 43%
> of flags. Different distribution, different partnerships -- school
> districts, not pantries.
>
> Then the **equity caveat**, which I want to be explicit about.
> **P3**, working educated households, are 37% of the test population
> but **less than half a percent of top-decile flags.** The model
> basically ignores them. That doesn't mean they're not food insecure
> -- it means the model can't see them. The recommendation there is
> **qualitative outreach**, not algorithmic targeting. Same for
> **P4**, elderly on SNAP -- monitor them through existing partnerships;
> don't take a low risk score as a green light.
>
> And the disclaimer: every coefficient is an **association**, not a
> causal claim. Survey weights and external county-level data are the
> obvious next steps.

---

## Slide 14 -- The Ask + Q&A  *(~30 s + Q&A)*

> Repository, code, every figure -- all reproducible from the four
> Jupyter notebooks. `random_state` 42 throughout. One `pip install`
> and one `jupyter nbconvert` away.
>
> Happy to take questions.

---

## Anticipated Questions and Prepared Answers

These are not on the slides; keep them in your back pocket.

### Q1. Why did logistic regression beat random forest?

> Two reasons. First, the gap is meaningful -- about 0.04 PR-AUC --
> but not huge. Second, my feature engineering already linearizes
> the strongest nonlinear effects: I added log-transformed distances
> and household-composition ratios. So the random forest's main
> advantage -- automatic interaction discovery -- is partly absorbed
> by the features the LR sees.

### Q2. Isn't `foodpantry` a leaky predictor of food insecurity?

> Honest answer: it's borderline, and I went back and forth on it.
> The data dictionary classifies it under "program participation",
> which suggests the case authors intend it as a feature. It
> contributes a meaningful but not dominant coefficient. If a real
> food bank deployed this, they'd want to think about whether to
> include it depending on whether they want to identify *new*
> households or just confirm pantry-using ones.

### Q3. Your silhouette score for the geographic clustering was 0.22. Isn't that quite low?

> Yes, and I want to be honest about it. Silhouette of 0.22 means
> the clusters are real but not crisply separated -- there's overlap
> in the boundary zone between rural and non-rural. The rural / non-rural
> split is *the* dominant axis but it's not a clean partition. That's
> also why I ran the geography-free clustering -- it gives a more
> nuanced picture even if its silhouette is similarly modest.

### Q4. Why K-Means rather than a Gaussian mixture model?

> K-Means was on the syllabus, scales well, and the silhouette
> diagnostics give me an honest answer about cluster separation.
> A GMM would let me capture elliptical clusters and would give soft
> assignments, which is genuinely useful for the "households at the
> boundary between two personas" question. That's a future-work item.

### Q5. Why didn't you use SMOTE?

> Time. The class imbalance here is moderate -- 28% positive, not
> severe. I judged that comparing the three other strategies --
> none, balanced weights, threshold tuning -- would be more
> informative within my budget than adding a fourth. SMOTE is
> first on the future-work list.

### Q6. How would you adapt this if a food bank had a hard budget for, say, 200 outreach contacts per month?

> The precision-recall-vs-cutoff curve is exactly the tool. You'd
> find the cutoff that produces 200 expected flags at the food
> bank's current population, look at the precision at that cutoff
> to estimate how many of the 200 contacts will reach truly
> food-insecure households, and look at the persona mix at that
> cutoff to plan staffing. Everything is in the synthesis notebook.

### Q7. The personas look interpretable, but how stable are they across data splits?

> Two pieces of evidence. First, on the chosen k, eight different
> random initializations of K-Means produce clusterings whose
> mean pairwise adjusted Rand index is above 0.9 -- so the
> algorithm is finding the same clusters consistently. I didn't
> run a full bootstrap of *the data* itself, which would be the
> gold standard. That's the next thing I'd do to validate that
> the personas aren't a sample artefact.

### Q8. What's your single recommended operating point?

> If forced to pick one number for a presentation slide: **target
> the top decile.** Precision 67%, recall 24%, persona composition
> dominated by P1 and P2. If they want broader coverage, the top
> quintile gets recall to 42% at the cost of dropping precision to
> 58% -- and the persona mix stays similar.

---

## Pacing Plan (12-15 min target)

| Slide | Section            | Time | Cumulative |
|-------|--------------------|------|------------|
| 1     | Title              | 0:20 | 0:20       |
| 2     | Hook               | 1:15 | 1:35       |
| 3     | Problem            | 0:45 | 2:20       |
| 4     | Data               | 1:00 | 3:20       |
| 5     | EDA insight        | 1:00 | 4:20       |
| 6     | Approach           | 0:50 | 5:10       |
| 7     | Track 1 result     | 1:15 | 6:25       |
| 8     | Operating point    | 0:50 | 7:15       |
| 9     | Drivers            | 1:00 | 8:15       |
| 10    | Track 2 discovery  | 1:00 | 9:15       |
| 11    | Personas           | 1:15 | 10:30      |
| 12    | Climax             | 1:15 | 11:45      |
| 13    | Recommendation     | 1:15 | 13:00      |
| 14    | Ask + Q&A          | 0:30 | 13:30      |

If you need to compress to **8-10 minutes**:

- Drop slide 8 (mention "well-calibrated, see the report").
- Trim slide 9 to a single sentence.
- Cut slide 11 to walking through P1, P2, and one of P3 or P4
  (not all five).
- Keep slides 2, 7, 12, 13 -- those are the load-bearing slides.
