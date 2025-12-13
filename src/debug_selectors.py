"""
Module: debug_selectors.py
Description: Utility script to reverse-engineer the HTML structure of the target website.
It searches for a known string (e.g., 'Toulouse') to identify the parent container class.
"""

import requests
from bs4 import BeautifulSoup

URL = "https://www.allrugby.com/competitions/top-14-2024/calendrier.html"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
TARGET_KEYWORD = "Toulouse"

def inspect_dom_structure():
    print(f"[DEBUG] Fetching {URL}...")
    response = requests.get(URL, headers=HEADERS)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Search for the text node containing the keyword
    element = soup.find(string=lambda text: text and TARGET_KEYWORD in text)

    if element:
        print(f"\n[SUCCESS] Keyword '{TARGET_KEYWORD}' found.")
        print("-" * 50)
        
        # Traverse up the DOM tree
        parent = element.parent
        grandparent = parent.parent
        great_grandparent = grandparent.parent
        
        print(f"1. Immediate Parent Tag:   <{parent.name} class='{parent.get('class')}>")
        print(f"2. Grandparent Tag:        <{grandparent.name} class='{grandparent.get('class')}>")
        print(f"3. Root Container Tag:     <{great_grandparent.name} class='{great_grandparent.get('class')}>")
        print("-" * 50)
        print("Use one of these classes in 'src/scraping.py' as MATCH_SELECTOR.")
    else:
        print(f"[FAILURE] Keyword '{TARGET_KEYWORD}' not found in the page source.")

if __name__ == "__main__":
    inspect_dom_structure()