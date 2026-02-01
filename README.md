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

## Notes
This repo intentionally avoids documenting competition rules here; see the Kaggle page for official details.
