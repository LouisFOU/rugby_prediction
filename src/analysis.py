"""
Module: analysis.py
Description: Performs basic exploratory data analysis (EDA) on the Clean Top 14 dataset.
Calculates the 'Home Field Advantage' metric and prepares the dataset for modeling.
"""

import pandas as pd
import os
import sys

# Constants
INPUT_FILE = "data/processed/top14_matches_2024.csv"
OUTPUT_FILE = "data/processed/dataset_modeling.csv"

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the CSV file into a pandas DataFrame.
    Raises FileNotFoundError if the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    return pd.read_csv(file_path)

def calculate_home_advantage(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a binary target column 'home_win' (1 if Home Score > Away Score, else 0).
    Computes and prints the global home win rate.
    """
    # Create target variable: 1 for Home Win, 0 for Draw/Away Win
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    
    total_matches = len(df)
    home_wins = df['home_win'].sum()
    win_rate = (home_wins / total_matches) * 100 if total_matches > 0 else 0
    
    print("-" * 30)
    print("ANALYSIS REPORT")
    print("-" * 30)
    print(f"Total Matches Analyzed : {total_matches}")
    print(f"Home Wins              : {home_wins}")
    print(f"Home Win Rate          : {win_rate:.2f}%")
    print("-" * 30)
    
    return df

def main():
    try:
        print(f"[INFO] Loading data from {INPUT_FILE}...")
        df = load_data(INPUT_FILE)
        
        df_processed = calculate_home_advantage(df)
        
        print(f"[INFO] Saving modeling dataset to {OUTPUT_FILE}...")
        df_processed.to_csv(OUTPUT_FILE, index=False)
        print("[SUCCESS] Process completed.")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()