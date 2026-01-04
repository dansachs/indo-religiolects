# Project Structure

This document describes the organized structure of the Religious Text Scraper project.

## Directory Structure

```
religiolect_model_V2/
├── src/                          # Core crawler source code
│   ├── config.py                # Configuration settings
│   ├── crawler.py               # Main crawler logic
│   ├── dashboard.py             # Live terminal dashboard
│   ├── nlp_processor.py         # NLP sentence extraction
│   └── run_pipeline.py          # Pipeline orchestration
│
├── data/                         # All data files
│   ├── scraped/                 # Individual religion CSV files
│   │   ├── Scraped_Catholic.csv
│   │   ├── Scraped_Islam.csv
│   │   ├── Scraped_Protestant.csv
│   │   └── Rejected_Non_Indonesian.csv
│   ├── combined/                 # Combined datasets
│   │   └── religious_corpus_*.csv
│   └── crawler_state/            # Crawler state files
│       ├── queue.json
│       ├── history.log
│       ├── content_hashes.txt
│       ├── crawler_stats.json
│       └── depth_boundary_urls.json
│
├── scripts/                      # Utility scripts
│   ├── combine_scraped_data.py  # Combine CSVs into single file
│   └── use_boundary_urls.py     # Add boundary URLs to seeds
│
├── training/                     # Model training scripts
│   ├── train_model.py           # Local training script
│   ├── train_model_colab.ipynb  # Google Colab notebook
│   └── requirements_training.txt
│
├── models/                       # Trained models (created during training)
│   └── trained/                 # Saved model checkpoints
│
├── config/                      # Configuration files
│   └── seeds.json               # Seed URLs for crawling
│
├── docs/                        # Documentation
│   └── COLAB_SETUP.md          # Colab setup guide
│
├── archive/                     # Archived old data
│
├── run_crawler.py              # Main entry point (run from root)
├── requirements.txt            # Python dependencies
└── README.md                   # Main documentation
```

## Running the Crawler

From the project root:

```bash
python run_crawler.py [options]
```

Or directly:

```bash
python src/run_pipeline.py [options]
```

## Running Scripts

```bash
# Combine scraped data
python scripts/combine_scraped_data.py

# Use boundary URLs
python scripts/use_boundary_urls.py
```

## Training Models

```bash
# Local training
python training/train_model.py

# Or use the Colab notebook
# Upload training/train_model_colab.ipynb to Google Colab
```

## File Locations

- **Scraped data**: `data/scraped/`
- **Combined datasets**: `data/combined/`
- **Crawler state**: `data/crawler_state/`
- **Seed URLs**: `config/seeds.json`
- **Trained models**: `models/trained/`

