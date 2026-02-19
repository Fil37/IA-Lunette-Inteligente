import os
from pathlib import Path

# --- CONFIGURATION ---
img_dir = r"P:\Videos\projet\Sans_Passage_ni_Feu"
out_labels_dir = r"P:\Videos\projet\Sans_Passage_ni_Feu\labels"

# Création du dossier de sortie s'il n'existe pas
os.makedirs(out_labels_dir, exist_ok=True)

# Extensions d'images à chercher
extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

print(f"--- Création des labels vides dans : {out_labels_dir} ---")

count = 0
for img_file in os.listdir(img_dir):
    if img_file.lower().endswith(extensions):
        # On récupère le nom sans l'extension (ex: image1.jpg -> image1)
        filename = Path(img_file).stem
        
        # On crée le fichier .txt vide
        with open(os.path.join(out_labels_dir, f"{filename}.txt"), "w") as f:
            pass  # On ne définit rien, donc le fichier reste vide
        
        count += 1

print(f"Terminé ! {count} fichiers .txt vides ont été générés.")