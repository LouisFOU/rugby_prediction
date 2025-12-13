"""
Module: scraping.py
Description: Scrapes Top 14 rugby match results from AllRugby.com.
Target URL: https://www.allrugby.com/competitions/top-14-2024/calendrier.html
Author: Louis Foujols
Date: 2025
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import sys

# --- CONFIGURATION ---
BASE_URL = "https://www.allrugby.com/competitions/top-14-2024/calendrier.html"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# TODO: This selector must be updated after running debug_selectors.py
MATCH_SELECTOR = '.mat'

def fetch_html_content(url: str) -> BeautifulSoup:
    """
    Fetches the HTML content from the given URL using a secure User-Agent.
    Returns a BeautifulSoup object or None if the request fails.
    """
    try:
        print(f"[INFO] Connecting to {url}...")
        response = requests.get(url, headers=HEADERS, timeout=10)
        
        if response.status_code == 200:
            return BeautifulSoup(response.content, 'html.parser')
        else:
            print(f"[ERROR] Connection failed with status code: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"[CRITICAL] An error occurred during connection: {e}")
        return None

def parse_matches(soup: BeautifulSoup) -> pd.DataFrame:
    """
    Extracts match data from the BeautifulSoup object based on CSS selectors.
    """
    matches_data = []
    match_elements = soup.select(MATCH_SELECTOR)
    
    print(f"[INFO] Found {len(match_elements)} match blocks using selector: '{MATCH_SELECTOR}'")

    for match in match_elements:
        try:
            # Extraction logic (simplified for now, to be expanded)
            content = match.get_text(separator=" ", strip=True)
            matches_data.append({'raw_content': content})
        except Exception as e:
            print(f"[WARNING] Could not parse a match block: {e}")
            continue

    return pd.DataFrame(matches_data)

def save_data(df: pd.DataFrame, filename: str):
    """
    Saves the DataFrame to the 'data/raw' directory.
    """
    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"[SUCCESS] Data saved to {path}")

if __name__ == "__main__":
    print("--- STARTING SCRAPER ---")
    
    soup = fetch_html_content(BASE_URL)
    
    if soup:
        df_matches = parse_matches(soup)
        
        if not df_matches.empty:
            save_data(df_matches, "top14_matches_raw.csv")
            print(df_matches.head())
        else:
            print("[ERROR] No data extracted. Please check the CSS selectors.")
            print("Action required: Run 'src/debug_selectors.py' to find the correct class name.")