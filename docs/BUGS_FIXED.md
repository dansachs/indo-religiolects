# Bugs Fixed During Reorganization

## ✅ Fixed Issues

### 1. **Crawler CSV Paths** (`src/crawler.py`)
- **Bug**: `_load_initial_stats()` was using old paths (`BASE_DIR / 'Scraped_*.csv'`)
- **Fix**: Updated to use new paths (`BASE_DIR / 'data' / 'scraped' / 'Scraped_*.csv'`)
- **Impact**: Stats loading from existing CSV files now works correctly

### 2. **Generic Output Path** (`src/run_pipeline.py`)
- **Bug**: Fallback path for unknown religions used old location
- **Fix**: Updated to `BASE_DIR / 'data' / 'scraped' / f'Scraped_{religion}.csv'` with directory creation
- **Impact**: Unknown religions will save to correct location

### 3. **Missing STATS_FILE Import** (`src/run_pipeline.py`)
- **Bug**: `STATS_FILE` was not imported but used in `--fresh` flag
- **Fix**: Added `STATS_FILE` to imports and archive list
- **Impact**: `--fresh` flag now properly archives stats file

### 4. **Training Script Missing Best Model Selection** (`training/train_model.py`)
- **Bug**: `load_best_model_at_end=True` without `metric_for_best_model` parameter
- **Fix**: Added `metric_for_best_model="f1"` and `greater_is_better=True`
- **Impact**: Training now properly selects best model based on F1 score

### 5. **Training Script Missing Learning Rate** (`training/train_model.py`)
- **Bug**: No explicit learning rate set (relies on default)
- **Fix**: Added `learning_rate=2e-5` (standard for BERT fine-tuning)
- **Impact**: More consistent training behavior

### 6. **Training Script Missing Training Info** (`training/train_model.py`)
- **Bug**: No training metadata saved
- **Fix**: Added `training_info.json` with model config and dataset info
- **Impact**: Better model tracking and reproducibility

## ✅ Verified Working

- ✅ All config paths resolve correctly
- ✅ CSV files are in correct locations
- ✅ Training script can find combined datasets
- ✅ All imports work correctly
- ✅ Archive functionality includes all state files

## ⚠️ Linter Warnings (Safe to Ignore)

The training script shows import warnings for:
- `pandas`, `sklearn`, `seaborn`, `matplotlib`, `datasets`, `transformers`, `torch`

These are **expected** - these packages are only installed in the Colab environment or training virtual environment, not in the base project. The script will work fine when run in the correct environment.

