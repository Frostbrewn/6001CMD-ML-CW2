# 6001CMD Machine Learning CW2

Group classification project for module 6001CMD. This project uses the UCI Bank Marketing dataset to predict whether a customer will subscribe to a term deposit.

## Project contents

| Path | Description |
|---|---|
| `data/raw/` | Local copy of the original dataset. This folder is excluded from Git. |
| `data/processed/` | Location for an exported processed dataset, if one is saved for submission. |
| `notebooks/01_dataset_audit.ipynb` | Initial dataset checks. |
| `notebooks/02_exploratory_data_analysis.ipynb` | Exploratory data analysis and figures. |
| `notebooks/03_shared_preprocessing_validation.ipynb` | Checks the shared preprocessing and data split. |
| `notebooks/02_base_models.ipynb` | Baseline and tuned Logistic Regression, Decision Tree and Random Forest models. |
| `notebooks/03_ensemble_xinyi.ipynb` | Integrated ensemble selection and final model evaluation. |
| `notebooks/04_independent_integration_xinyi.ipynb` | Separate integration prototype. It is not the main final workflow. |
| `src/shared_preprocessing.py` | Shared feature preparation, data split and preprocessing functions. |
| `src/ensemble.py` | Soft Voting and Stacking model helpers. |
| `src/evaluation_plots.py` | Model comparison, confusion matrix and ROC curve plots. |
| `results/xinyi_final/` | Saved model metrics, experiment notes and evaluation figures. |
| `requirements.txt` | Pinned Python package versions. |

## Requirements

- Python 3.10.x
- Git
- Visual Studio Code
- The VS Code Python and Jupyter extensions

The project dependencies are listed in `requirements.txt`.

## 1. Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/Frostbrewn/6001CMD-ML-CW2.git
cd 6001CMD-ML-CW2
```

## 2. Create and activate the virtual environment

### Windows PowerShell

Create the environment:

```powershell
py -3.10 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks the activation script, allow scripts for the current terminal session only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The terminal prompt should show `(.venv)` when the environment is active.

### macOS, Linux or GitHub Codespaces

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install the project dependencies

With the virtual environment active, run:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

To register the environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name 6001cmd-ml-cw2 --display-name "Python (.venv) - 6001CMD ML CW2"
```

In Visual Studio Code, open a notebook, select **Kernel**, and choose **Python (.venv) - 6001CMD ML CW2**. To select the interpreter for the project, open the Command Palette with `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and select the Python executable inside `.venv`.

## 4. Download the dataset

The project uses the original `bank-full.csv` file from the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing).

The original dataset is not stored in this repository. Each team member must download it separately.

1. Open the [official UCI dataset page](https://archive.ics.uci.edu/dataset/222/bank+marketing) and download the dataset archive. The [direct ZIP download](https://archive.ics.uci.edu/static/public/222/bank+marketing.zip) is also available.
2. Extract the downloaded archive. It contains `bank.zip` and `bank-additional.zip`.
3. Extract **`bank.zip`**, not `bank-additional.zip`. The project uses the older `bank-full.csv` file with 45,211 records and 17 columns.
4. Create the following folder in the repository if it does not already exist:
   `data/raw/`
5. Copy `bank-full.csv` into that folder.

The final file location must be:

```text
6001CMD-ML-CW2/
└── data/
    └── raw/
        └── bank-full.csv
```

On Windows PowerShell, the folder can be created with:

```powershell
New-Item -ItemType Directory -Force .\data\raw
```

Check that the file is in the correct location:

```powershell
Test-Path .\data\raw\bank-full.csv
```

The command should return `True`.

The notebooks and shared preprocessing module expect this exact filename and location. The CSV uses a semicolon separator, which is handled by the shared data-loading function.

### Dataset licence and attribution

The UCI Bank Marketing dataset is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Credit the dataset source when reusing or sharing it.

Dataset citation:

Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

## 5. Run the notebooks

Run the notebooks from the repository root, using the project virtual environment.

Recommended workflow:

1. `01_dataset_audit.ipynb` checks the original dataset.
2. `02_exploratory_data_analysis.ipynb` examines the data and produces the EDA figures.
3. `03_shared_preprocessing_validation.ipynb` checks the shared feature preparation, train/test split and cross-validation setup.
4. `02_base_models.ipynb` compares and tunes the three base classifiers using training cross-validation.
5. `03_ensemble_xinyi.ipynb` selects the two base models and ensemble method using training cross-validation, then performs the final test evaluation.

The notebook `04_independent_integration_xinyi.ipynb` is a separate prototype and is not required for the main workflow.

## 6. Data handling

The original dataset must stay in `data/raw/` on each team member's machine. The `.gitignore` file excludes this folder, so the raw CSV should not be committed or pushed to GitHub.

The shared preparation function in `src/shared_preprocessing.py` creates the model input features and target, including the agreed feature engineering. The preprocessing pipelines are fitted during model training and cross-validation.

A processed CSV is not currently stored in the repository. If an exported processed dataset is needed, save it under `data/processed/`. That folder is not excluded by `.gitignore`, so files saved there can be committed. Do not place the original raw CSV in that folder.

## 7. Reproducibility notes

- The shared random state is `42`.
- The train/test split is stratified and uses an 80:20 ratio.
- Model selection and hyperparameter tuning use five-fold stratified cross-validation.
- The positive target class is encoded as `1` (`yes`), and the negative class as `0` (`no`).
- The `duration` feature is excluded from model inputs because it is only known after the call.
- Preprocessing is included in the model pipelines so that transformations are fitted within each training fold.
