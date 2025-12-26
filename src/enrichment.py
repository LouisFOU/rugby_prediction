"""
Module: enrichment.py
Description: Adds Stadium, Budget, and REAL International Window detection.
"""

import pandas as pd
import os

INPUT_FILE = "data/processed/top14_historical_matches.csv"
OUTPUT_FILE = "data/processed/dataset_enriched.csv"

# Handmade Metadata (Budgets Tiers & Stades)
CLUB_METADATA = {
    'Toulouse':    {'budget': 10, 'capacity': 19000},
    'Paris':       {'budget': 10,  'capacity': 20000},
    'Clermont':    {'budget': 7,  'capacity': 19000}, 
    'Toulon':      {'budget': 7,  'capacity': 18000},
    'La Rochelle': {'budget': 7,  'capacity': 16000},
    'Bordeaux':    {'budget': 7,  'capacity': 33000},
    'Racing 92':   {'budget': 6,  'capacity': 30000},
    'Montpellier': {'budget': 6,  'capacity': 15600}, 

    'Lyon':        {'budget': 7,  'capacity': 25000},
    'Castres':     {'budget': 5,  'capacity': 12500},
    'Pau':         {'budget': 5,  'capacity': 18000},
    'Bayonne':     {'budget': 5,  'capacity': 16900},
    
    'Perpignan':   {'budget': 3,  'capacity': 14500},
}


def enrich_data():
    if not os.path.exists(INPUT_FILE):
        print("Input file missing.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Enriching {len(df)} matches...")

    # Helper pour sécuriser les données manquantes
    def get_meta(team, key):
        return CLUB_METADATA.get(team, {'budget': 2, 'capacity': 12000})[key]

    df['home_budget'] = df['home_team'].apply(lambda x: get_meta(x, 'budget'))
    df['away_budget'] = df['away_team'].apply(lambda x: get_meta(x, 'budget'))
    df['stadium_capacity'] = df['home_team'].apply(lambda x: get_meta(x, 'capacity'))

    df['score_diff'] = df['home_score'] - df['away_score']
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset enriched. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    enrich_data()