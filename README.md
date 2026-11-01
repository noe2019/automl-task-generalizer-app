# AutoML Task Generalizer

Profiles a dataset, detects the task (classification or regression), trains and compares models with cross-validation and exports the best model for deployment.

**Live app:** https://noe-automl-55f69a9f06c9.herokuapp.com/
**Portfolio:** https://noe2019.github.io/DA-Portfolio-Website/

## Features
- CSV upload or sample datasets
- Automatic data profiling
- Task detection and model leaderboard (5-fold CV)
- Download of the best fitted pipeline (.joblib)

## Tech stack
Python, scikit-learn, Streamlit

## Data
All data in this app is **simulated** (generated in `data.py` with a fixed random seed) to reproduce the structure of the original project without exposing real or confidential records.

The original version used PyCaret. This version uses a pure scikit-learn AutoML loop so it installs fast and fits Heroku's slug and memory limits.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment
Deployed on Heroku (stack `heroku-24`, Python 3.12) using the `Procfile` in this repository.

---
Author: Noé Carème Fouotsa Manfouo
