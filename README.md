# MLOPS_PROJECT_1

End-to-end classification pipeline for hotel booking cancellations, built with Python, LightGBM, and MLflow.

## Project overview

The pipeline performs:
- Data ingestion from Google Cloud Storage (`bucket_name`, `bucket_file_name`)
- Train/test split (`train_ratio` from `config/config.yaml`)
- Data preprocessing: drop irrelevant columns, dedupe, encoding, skew handling
- SMOTE balancing and feature selection using RandomForest
- Hyperparameter tuning with `RandomizedSearchCV` on `LGBMClassifier`
- Model evaluation (accuracy, precision, recall, f1)
- Model serialization (`artifacts/model/lightgbm.pkl`) and MLflow artifact logging

## Repository structure

- `artifacts/` - generated data and model outputs
  - `raw/` - downloaded raw CSV + train/test split
  - `processed/` - preprocessed train/test sets
  - `model/` - serialized model checkpoint
- `config/` - YAML and Python parameter definitions
  - `config.yaml` - ingestion + processing settings
  - `model_params.py` - hyperparameter search space
  - `paths_config.py` - file path constants
- `pipeline/` - orchestrator script:
  - `training_pipeline.py`
- `src/` - core processing modules
  - `data_ingestion.py`
  - `data_preprocess.py`
  - `model_training.py`
  - `logger.py`
  - `custom_exception.py`
- `utils/` - utility helpers:
  - `comman_functions.py` (read_yaml, load_data)
- `notebook/` - exploratory notebook + sample data
- `requirements.txt`, `setup.py`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

- `config/paths_config.py` defines locations for raw/processed/model files and `CONFIG_PATH`.
- `config/config.yaml` controls GCS download settings and preprocessing columns.
- `config/model_params.py` defines the LightGBM search space and `RandomizedSearchCV` settings.

## Usage

1. Update `config/config.yaml` with GCS credentials accessible in the environment.
2. Ensure GCP auth is set (e.g. `GOOGLE_APPLICATION_CREDENTIALS`).
3. Run pipeline:

```bash
python -m pipeline.training_pipeline
```

4. Outputs:
- `artifacts/raw/train.csv`, `artifacts/raw/test.csv`
- `artifacts/processed/processed_train.csv`, `.../processed_test.csv`
- `artifacts/model/lightgbm.pkl`
- `logs/log_YYYY-MM-DD.log`
- MLflow run in current working directory (local tracking URI default)

## File specifics

- `src/data_ingestion.py`:
  - Downloads from GCS and performs train/test split
- `src/data_preprocess.py`:
  - Drops `Unnamed: 0` and `Booking_ID`
  - Label encodes categorical cols from config
  - Applies log transform on skewed numerical cols (>skewness_threshold)
  - SMOTE balancing and feature selection (top `no_of_features`)
- `src/model_training.py`:
  - Loads processed data, fits LightGBM with randomized search, evaluates, saves
  - Logs params/metrics/artifacts with MLflow

## Notes & best practices

- Validate CSV schema: expected columns include `booking_status`, booking features, etc.
- Ensure `booking_status` target exists and is binary for metrics to operate.
- For reproducible hyperparams, adjust `n_iter`, `cv`, and set seeds in `config/model_params.py`.

## Troubleshooting

- GCP download fails: check service account permissions and bucket/object path.
- MLflow errors: install and configure MLflow or disable by commenting in `ModelTraining.run()`.
- Data schema mismatch: align raw data to columns listed in `config/config.yaml`.

## License

MIT
