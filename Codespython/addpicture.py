# -*- coding: utf-8 -*-
import hashlib
import os
import shutil
from pathlib import Path

def calculate_hash(image_path):
    hasher = hashlib.md5()
    with open(image_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def clean_and_copy_dataset(source_dir, current_dataset_dir, output_dir):
    # 1. Scanner le dataset actuel pour connaître les images déjà connues
    existing_hashes = set()
    print(f"--- Phase 1 : Indexation du dataset actuel ({current_dataset_dir}) ---")
    
    # On vérifie que le dossier existe avant de scanner
    if os.path.exists(current_dataset_dir):
        for img_path in Path(current_dataset_dir).rglob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                existing_hashes.add(calculate_hash(img_path))
        print(f"Images deja connues : {len(existing_hashes)}")
    else:
        print("Dossier dataset existant vide ou introuvable. Indexation ignoree.")

    # 2. Scanner les nouveaux dossiers et copier uniquement l'inédit
    print(f"\n--- Phase 2 : Filtrage et copie depuis '{source_dir}' ---")
    count_added = 0
    count_skipped = 0

    if not os.path.exists(source_dir):
        print(f"ERREUR: Le dossier source '{source_dir}' n'existe pas !")
        return

    for img_path in Path(source_dir).rglob('*'):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                img_hash = calculate_hash(img_path)
                
                if img_hash not in existing_hashes:
                    # On recrée l'arborescence
                    relative_path = img_path.relative_to(source_dir)
                    dest_path = Path(output_dir) / relative_path
                    
                    # Créer le dossier de destination si besoin
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copier le fichier
                    shutil.copy2(img_path, dest_path)
                    
                    # Ajout au set pour éviter les doublons dans le nouveau lot lui-même
                    existing_hashes.add(img_hash) 
                    count_added += 1
                    print(f"[AJOUT] {relative_path}")
                else:
                    count_skipped += 1
            except Exception as e:
                print(f"Erreur sur l'image {img_path}: {e}")

    print(f"\n--- Bilan ---")
    print(f"Images ajoutees vers '{output_dir}' : {count_added}")
    print(f"Doublons ignores : {count_skipped}")

# --- CONFIGURATION ---
if __name__ == '__main__':
    # Utilise r"" pour éviter les erreurs de chemin Windows
    base_dir = r"P:\Videos\projet"
    
    # Adapte ces chemins selon ton arborescence réelle
    DOSS_NOUVELLES_IMAGES = os.path.join(base_dir, "Dataset", "Nouveau dossier")
    DATASET_EXISTANT = os.path.join(base_dir, "Dataset", "train", "images")
    DOSSIER_SORTIE = os.path.join(base_dir, "New_Picture")

    clean_and_copy_dataset(DOSS_NOUVELLES_IMAGES, DATASET_EXISTANT, DOSSIER_SORTIE)