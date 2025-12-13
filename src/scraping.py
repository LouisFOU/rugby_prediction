"""
Module: scraping.py
Description: Scrapes and cleans Top 14 rugby match results.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re # Import du module Regex (Expressions Régulières)

# --- CONFIGURATION ---
BASE_URL = "https://www.allrugby.com/competitions/top-14-2024/calendrier.html"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
MATCH_SELECTOR = '.mat' 

def fetch_html_content(url: str) -> BeautifulSoup:
    try:
        print(f"[INFO] Connecting to {url}...")
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            return BeautifulSoup(response.content, 'html.parser')
        else:
            print(f"[ERROR] Connection failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"[CRITICAL] Connection error: {e}")
        return None

def clean_match_data(raw_text: str) -> dict:
    """
    Parses a string like 'Bayonne 26 - 7 Toulouse détails' 
    into structured data: Home, Away, Scores.
    """
    # Regex Pattern explanation:
    # (.*?)   -> Capture Home Team (any text until the number)
    # (\d+)   -> Capture Home Score (digits)
    # \s*-\s* -> Ignore the separator " - "
    # (\d+)   -> Capture Away Score (digits)
    # (.*)    -> Capture Away Team (rest of the text)
    pattern = r"(.*?) (\d+)\s*-\s*(\d+) (.*)"
    
    match = re.search(pattern, raw_text)
    
    if match:
        home_team = match.group(1).strip()
        # Remove 'détails' from the away team name if present
        away_team = match.group(4).replace("détails", "").strip()
        
        return {
            'home_team': home_team,
            'home_score': int(match.group(2)),
            'away_score': int(match.group(3)),
            'away_team': away_team
        }
    return None

def parse_matches(soup: BeautifulSoup) -> pd.DataFrame:
    matches_data = []
    match_elements = soup.select(MATCH_SELECTOR)
    
    print(f"[INFO] Parsing {len(match_elements)} raw blocks...")

    for match in match_elements:
        try:
            content = match.get_text(separator=" ", strip=True)
            cleaned = clean_match_data(content)
            
            if cleaned:
                matches_data.append(cleaned)
                
        except Exception as e:
            continue

    return pd.DataFrame(matches_data)

def save_data(df: pd.DataFrame, filename: str):
    output_dir = "data/processed" # On change de dossier car c'est de la donnée propre
    os.makedirs(output_dir, exist_ok=True)
    
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"[SUCCESS] Clean data saved to {path}")

if __name__ == "__main__":
    soup = fetch_html_content(BASE_URL)
    if soup:
        df_matches = parse_matches(soup)
        if not df_matches.empty:
            print(df_matches.head())
            save_data(df_matches, "top14_matches_2024.csv")
        else:
            print("[ERROR] No data extracted.")