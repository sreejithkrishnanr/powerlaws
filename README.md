PowerLaws
==============================

Power Laws: Forecasting Energy Consumption Competition - drivendata.org

Introduction
------------

There are four types of models used in this project.

1. XGB model: In order to prediction consumption at time t, this model uses features at t-output_window_size. This approach is simple and the forecast for entire prediction window is made without using consumtion predicted for previous timesteps
2. Log transformed model: This is same as XGB model except consumption values are log transformed
3. Stationary model: This is same as XGB model except consumption values are made stationary using moving average
4. Recursive model: This model uses features at timestep t-1 to make predictions for timestep t

For each site, seperate models of above types are trained and evaluated. Model with best score is used for prediction.


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models and model summaries
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to prepare raw data
    │   │   └── make_holidays.py            <- Script to prepare raw holidays.csv
    |   |   └── make_split_data_by_freq.py  <- Script to split train / test data according to frequency
    |   |   └── make_submission.py          <- Script to generate submission in proper format from generated predictions for different frequencies
    |   |   └── make_weather.py             <- Script to prepare raw weather.csv
    │   │
    │   ├── features       <- Scripts to turn raw/interim data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── model_common.py     <- Feature selection, model registry
    │   │   ├── model_hparams.py    <- Hyperparameter for all models
    │   │   ├── model_gb.py         <- XGB model, uses features at t-output_window_size to predict consumption at time t
    │   │   ├── model_gb_log.py     <- Same as model_gb.py except consumption values are log transformed
    │   │   ├── model_stationary.py <- Same as model_gb.py except consumption values are made stationary using moving average
    │   │   ├── model_recursive.py  <- XGB model, uses features at t-1 to predict consumption at t
    │   │   ├── model_rnn.py        <- RNN model
    │   │   ├── train_model.py      <- Train and score separate models for each site
    │   │   └── predict_model.py    <- Predict using best model for specific site
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


Setup
------------

### Prerequisites
Python 3 and GNU make

### Recommended system
Atleast 16GB RAM with ~20 GB free disk space. Tested using Mac OS.

### Installing dependencies
Run `pip install -r requirements.txt`

### Data setup
Put holidays.csv, metadata.csv, submission_format.csv, submission_frequency.csv, train.csv and weather.csv in data/raw folder

Running
------------

Run `make all`. Predictions will be stored in `predictions/submission.csv`. Models for each site will be stored in `models/` folder.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
