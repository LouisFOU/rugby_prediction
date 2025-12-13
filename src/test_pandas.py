import pandas as pd

print("1. Démarrage de Pandas...")

# Chemin vers tes données propres
csv_path = "data/processed/top14_matches_2024.csv"

try:
    print(f"2. Lecture du fichier {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print("\n✅ SUCCÈS ! Fichier chargé.")
    print(f"Dimensions : {df.shape} (Lignes, Colonnes)")
    print("Aperçu des 3 premières lignes :")
    print(df.head(3))
    
except FileNotFoundError:
    print("❌ ERREUR : Le fichier CSV n'est pas trouvé. Vérifie le chemin.")
except Exception as e:
    print(f"❌ ERREUR CRITIQUE : {e}")
    