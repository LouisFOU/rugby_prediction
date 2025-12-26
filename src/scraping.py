"""
Module: scraping.py
Description: Scrapes Top 14 rugby match results (2014-2026).
Features: Anti-bot delays (2seconds), Date extraction (Day/Month) for international window detection.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import re
import time
import random

BASE_URL_TEMPLATE = "https://www.allrugby.com/competitions/top-14-{}/calendrier.html"
START_YEAR = 2015
END_YEAR = 2026 

#J'ai mis mon header propre pour que le site ne bloque pas mon scraping
HEADERS = {

    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

}

def fetch_html_content(url: str) -> BeautifulSoup:
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        if response.status_code == 200:
            return BeautifulSoup(response.content, 'html.parser')
        return None
    except Exception as e:
        print("Connection error")
        return None


def clean_match_data(raw_text: str, year: int, month: int) -> dict:
    pattern = r"(.*?) (\d+)\s*-\s*(\d+) (.*)"
    match = re.search(pattern, raw_text)
    
    if match:
        return {
            'season': year,
            'home_team': match.group(1).strip(),
            'home_score': int(match.group(2)),
            'away_score': int(match.group(3)),
            'away_team': match.group(4).replace("détails", "").strip()
        }
    return None

def parse_season(year: int) -> pd.DataFrame:
    url = BASE_URL_TEMPLATE.format(year)
    soup = fetch_html_content(url)
    
    if not soup:
        return pd.DataFrame()

    matches_data = []
    
    #Il a fallu regarder le html du site allrugby
    # On cherche tous les H3 (dates potentielles) et les .mat (matchs)
    # L'ordre est important : on lit la page de haut en bas
    elements = soup.find_all(['h3', 'a'])
    
    current_month = 0
    
    for el in elements:
        # Si c'est un titre (Date potentielle)
        if el.name == 'h3':
            month = parse_date(el.get_text(strip=True))
            if month:
                current_month = month
                
        # Si c'est un match (classe .mat)
        elif el.name == 'a' and 'mat' in el.get('class', []):
            content = el.get_text(separator=" ", strip=True)
            cleaned = clean_match_data(content, year, current_month)
            if cleaned:
                matches_data.append(cleaned)

    print(f"Season {year}: Found {len(matches_data)} matches.")
    return pd.DataFrame(matches_data)

def main():
    all_seasons_data = []

    for year in range(START_YEAR, END_YEAR + 1):
        df_season = parse_season(year)
        if not df_season.empty:
            all_seasons_data.append(df_season)
        
        # Pause pour ne pas se faire bannir du site pas utile ici (après test)
        

    if all_seasons_data:
        master_df = pd.concat(all_seasons_data, ignore_index=True)
        
        # Sauvegarde
        output_path = "data/processed/top14_historical_matches.csv"
        os.makedirs("data/processed", exist_ok=True)
        master_df.to_csv(output_path, index=False)
        
        print(f"Saved {len(master_df)} matches to {output_path}")
        print("Columns: season, month, home_team, ...")
    else:
        print("No data collected.")

if __name__ == "__main__":
    main()