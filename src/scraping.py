import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL cible : Résultats du Top 14 (Saison actuelle pour tester)
BASE_URL = "https://www.allrugby.com/competitions/top-14-2024/calendrier.html"

def check_connection():
    """
    Vérifie si on peut accéder au site sans être bloqué.
    """
    # Le User-Agent est CRUCIAL. Il fait croire au site qu'on est un navigateur Chrome et pas un robot.
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        print(f"📡 Tentative de connexion à {BASE_URL}...")
        response = requests.get(BASE_URL, headers=headers)
        
        # Code 200 = Succès (HTTP OK)
        if response.status_code == 200:
            print("✅ SUCCÈS : Connexion établie ! (Status 200)")
            return response
        else:
            print(f"❌ ÉCHEC : Le site a renvoyé le code {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE : {e}")
        return None

if __name__ == "__main__":
    check_connection()