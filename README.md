# js_rt_market_forecast

Work-in-progress implementation for a forecasting model. This README focuses on how this repo is organized and how to run the code you build here.

## Repo Layout
- `data/`: Kaggle data (ignored by git)
- `notebooks/`: EDA and experiments
- `src/`: model code, training, and inference utilities
- `models/`: saved models and artifacts (ignored by git)
- `outputs/`: metrics, logs, plots (ignored by git)
- `submissions/`: generated submissions (ignored by git)
- `requirements.md`: dependency list

## Setup
Dependencies are listed in `requirements.md`. Create a virtual environment and install the packages you need for your workflow.

## Workflow
1. Drop the competition data into `data/`.
2. Use `notebooks/` for exploration and feature ideas.
3. Implement reusable training/inference code in `src/`.
4. Write submissions to `submissions/`.

## Baseline (Local)
Use the JSON config to control splits and feature selection:

```bash
python -m src.train --config config/baseline.json --model-out models/ridge.pkl
python -m src.predict --config config/baseline.json --model models/ridge.pkl --output submissions/mock_submission.csv
```

## Models
Supported model names (see `src/models.py`):
- `ridge`
- `elasticnet`
- `hgb` (HistGradientBoosting)
- `rf` (RandomForest)
- `extra_trees`
- `lgbm` (LightGBM, optional dependency)
- `xgb` (XGBoost, optional dependency)
- `cat` (CatBoost, optional dependency)

You can pass model params as a JSON string or a JSON file path:

```bash
python -m src.train --config config/baseline.json --model hgb --model-params '{"max_iter": 500, "learning_rate": 0.05}'
```

## End-to-End
Train and generate a submission in one step:

```bash
python -m src.e2e --config config/baseline.json --model hgb --model-out models/hgb.pkl --output submissions/hgb_submission.csv
```

## Notes
This repo intentionally avoids documenting competition rules here; see the Kaggle page for official details.
