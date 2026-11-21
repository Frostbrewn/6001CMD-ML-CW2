# Xinyi's independent integration prototype — preliminary results

These results came from the independent prototype run on the full Bank Marketing dataset (45,211 records). They **are not final or team-approved results**. Please reconcile preprocessing, split and hyperparameters with Ben and Zyen before using them as official group findings.

- Data split: 80% train / 20% held-out test, stratified, random seed 42.
- Input column `duration` excluded for a pre-call prediction setting.
- Training CV selected Random Forest and Decision Tree to form two ensemble candidates.
- Soft Voting was selected over Stacking by training CV F1 (0.4637 versus 0.3359).
- On held-out test: Random Forest had F1 **0.4824** and ROC-AUC **0.7990**; Soft Voting had F1 **0.4729** and ROC-AUC **0.7911**.
- The ensemble did not outperform Random Forest on F1 or ROC-AUC in this experiment; do not claim it did.

Files:
- `training_cv.csv` — cross-validation model selection
- `test_comparison.csv` — held-out test metrics
- `experiment_notes.json` — independent experiment assumptions and ensemble choice

The original run also produced `model_comparison.png`, `confusion_matrices.png` and `roc_curves.png` locally; these binary images need to be added separately from the local machine. Do not commit `bank-full.csv` (the raw dataset).
