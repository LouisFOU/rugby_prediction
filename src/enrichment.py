"""
Module: enrichment.py
Description: Adds Stadium, Budget, and REAL International Window detection.
"""

import pandas as pd
import os

INPUT_FILE = "data/processed/top14_historical_matches.csv"
OUTPUT_FILE = "data/processed/dataset_enriched.csv"

# Métadonnées (Budgets Tiers & Stades)
CLUB_METADATA = {
    'Agen': {'budget': 1, 'capacity': 14000}, 'Bayonne': {'budget': 2, 'capacity': 16900},
    'Biarritz': {'budget': 1, 'capacity': 13500}, 'Bordeaux': {'budget': 3, 'capacity': 33000},
    'Brive': {'budget': 1, 'capacity': 13900}, 'Castres': {'budget': 2, 'capacity': 12500},
    'Clermont': {'budget': 3, 'capacity': 19000}, 'Grenoble': {'budget': 1, 'capacity': 20000},
    'La Rochelle': {'budget': 3, 'capacity': 16000}, 'Lyon': {'budget': 3, 'capacity': 25000},
    'Montpellier': {'budget': 3, 'capacity': 15600}, 'Mt.Marsan': {'budget': 1, 'capacity': 10000},
    'Oyonnax': {'budget': 1, 'capacity': 11400}, 'Paris': {'budget': 3, 'capacity': 20000},
    'Pau': {'budget': 2, 'capacity': 18000}, 'Perpignan': {'budget': 1, 'capacity': 14500},
    'Racing 92': {'budget': 3, 'capacity': 30000}, 'Toulon': {'budget': 3, 'capacity': 18000},
    'Toulouse': {'budget': 3, 'capacity': 19000}, 'Vannes': {'budget': 1, 'capacity': 11000}
}

def is_doublon(month: int) -> int:
    """
    1 if match is during Six Nations (Feb-Mar) or Autumn Tests (Nov).
    """
    if month in [2, 3, 11]:
        return 1
    return 0

def enrich_data():
    if not os.path.exists(INPUT_FILE):
        print("Input file missing.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"[INFO] Enriching {len(df)} matches...")

    # Helper pour sécuriser les données manquantes
    def get_meta(team, key):
        return CLUB_METADATA.get(team, {'budget': 1, 'capacity': 12000})[key]

    # 1. Budget & Stade
    df['home_budget'] = df['home_team'].apply(lambda x: get_meta(x, 'budget'))
    df['away_budget'] = df['away_team'].apply(lambda x: get_meta(x, 'budget'))
    df['stadium_capacity'] = df['home_team'].apply(lambda x: get_meta(x, 'capacity'))

    # 2. Doublons (Basé sur le mois récupéré par le scraper)
    df['is_international_window'] = df['month'].apply(is_doublon)
    
    # 3. Target
    df['score_diff'] = df['home_score'] - df['away_score']
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[SUCCESS] Dataset enriched. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    enrich_data()