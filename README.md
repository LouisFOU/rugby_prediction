# Top 14 Rugby Predictor (AI-Powered)

> A Machine Learning pipeline to predict the outcome of the French Rugby Championship (Top 14) for the 2026 season by Foujols Louis (https://www.linkedin.com/in/louis-foujols)

![Python](https://img.shields.io/badge/Python-3.11-blue) ![XGBoost](https://img.shields.io/badge/Model-XGBoost-green) ![Status](https://img.shields.io/badge/Status-Completed-success)

## 📋 Overview
This project applies **Data Science** and **Predictive Modeling** to sports analytics. By scraping over 10 years of historical match data, enriching it with economic and physical metadata (budgets, stadium capacity), and training an **XGBoost Classifier**, this engine simulates the entire 2026 Top 14 season.

It goes beyond simple win/loss prediction by calculating probabilistic outcomes to attribute **Offensive and Defensive Bonus Points**, which are critical in the French league system.

## ⚙️ Key Technical Features

### 1. Data Engineering (ETL)
- **Web Scraping:** Automated extraction of 2,000+ matches (2014-2025) using `BeautifulSoup`.
- **Data Enrichment:** Integration of club budgets, stadium capacities, and "International Window" detection (Six Nations tournament impact).
- **Proxy Handling:** Implemented a proxy mechanism to simulate promoted teams (e.g., Montauban) using statistical profiles of past promoted squads (e.g., Vannes).

### 2. Machine Learning Strategy
- **Algorithm:** Transitioned from Random Forest to **XGBoost** for better handling of tabular data and ranking.
- **Temporal Weighting:** Implemented a time-decay function so that recent matches (2024) have significantly more weight in training than older ones (2015).
- **Feature Engineering:**
  - `home_exp` / `away_exp`: Quantified "Playoff Experience" to model the psychological edge in high-stakes games.
  - `is_international_window`: Boolean flag to account for roster depletion during international tests.

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install -r requirements.txt
