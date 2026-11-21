# Xinyi — Independent integration findings (PROVISIONAL)

**Not official group results.** This is a separate end-to-end prototype to unblock Part F and Part G. Ben and Zyen must confirm the common preprocessing, prediction timing, train/test split and final tuning choices before these numbers enter the submitted report.

## Reproducible setup

- Dataset: UCI `bank-full.csv`, **45,211** observations
- Task: binary classification (`yes=1`, `no=0`)
- Split: **80% train (36,168)** / **20% test (9,043)**, stratified, random state 42
- Positive-class proportion in training: approximately 11.7%
- **Post-call `duration` is excluded** to model a *before-contact* deployment situation; this is a deliberate prototype assumption to confirm with Ben
- `unknown` categories are retained; `pdays=-1` remains a meaningful sentinel
- Numerical median imputation and standard scaling; categorical most-frequent imputation and one-hot encoding are **inside the model Pipeline**, preventing validation/test leakage
- Three Zyen-style algorithms with **small independent GridSearchCV grids** (3-fold stratified CV, optimizing positive-class F1); this is **not** a copy of Zyen's full official tuning experiment
- Two base models and the Voting-vs-Stacking technique are selected using training-only validation; the held-out test set is evaluated only after selection

## Training cross-validation results

| Base model | Best CV F1 |
|---|---:|
| Logistic Regression | 0.3723 |
| Decision Tree | 0.4164 |
| Random Forest | **0.4709** |

**Candidate base model pair:** Random Forest + Decision Tree (highest F1 in this independent grid). Logistic Regression was excluded *for this prototype only* due to its lower training-CV F1; the final team should additionally inspect diversity, recall, and practical constraints.

| Ensemble candidate | Training CV F1 |
|---|---:|
| Soft Voting | **0.4637** |
| Stacking | 0.3359 |

**Validation-selected approach:** Soft Voting. Stacking was considered but had worse training-CV F1 under the specified meta-classifier and folds.

## Held-out test performance — independent prototype only

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7549 | 0.2663 | 0.6238 | 0.3733 | 0.7722 |
| Decision Tree | 0.8212 | 0.3407 | 0.5652 | 0.4252 | 0.7630 |
| **Random Forest** | **0.8664** | **0.4412** | 0.5321 | **0.4824** | **0.7990** |
| Soft Voting Ensemble | 0.8550 | 0.4115 | 0.5558 | 0.4729 | 0.7911 |

The ensemble achieves marginally **lower test F1 and ROC-AUC than Random Forest alone** (F1: 0.4729 vs 0.4824), though it has slightly higher recall (0.5558 vs 0.5321). **Do not claim that the ensemble outperforms the best individual model.**

Under these provisional settings, **Random Forest** is the more defensible overall deployment candidate if F1 and ROC-AUC are priorities, while the ensemble has a modest recall advantage that might matter if missing potential subscribers is particularly costly. The group should select based on an agreed objective and deployment costs.

## Next steps and limitations

1. Ben confirms the deployment context (notably `duration`), approved preprocessing, and split.
2. Zyen shares final tuned models, hyperparameters and cross-validation results.
3. Replace this independent prototype with the shared official interface in `03_ensemble_xinyi.ipynb`.
4. Re-run final model comparisons **once** with the approved test set; interpret confusion matrices and graphs from that run.
5. Report computational costs, model limitations, generalisation and references; disclose AI assistance as required.

**Run locally:** `python src/independent_workflow_xinyi.py --data data/raw/bank-full.csv`

Local outputs are saved to `data/raw/xinyi_local_prototype/` and excluded from Git by the existing `.gitignore`.
