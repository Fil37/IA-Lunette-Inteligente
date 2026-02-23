import cv2
import numpy as np
import os
import shutil
import random
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
parent_dir = BASE_DIR.parent;
def ajuster_luminosite_aleatoire(image, mode="dark"):
    # Conversion vers l'espace HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Définition du facteur aléatoire selon le mode
    if mode == "dark":
        facteur = random.uniform(0.3, 0.8) # Assombrit de façon variable
    else:
        facteur = random.uniform(1.2, 1.8) # Éclaircit de façon variable

    # Application du facteur sur le canal V (Luminosité)
    v = np.array(v, dtype=np.float64)
    v = v * facteur
    
    # On bloque les valeurs entre 0 et 255
    v = np.clip(v, 0, 255).astype(np.uint8)

    # Recomposition
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def augment_luminosite_random(img_dir, label_dir, output_dir):
    out_images = Path(output_dir) / "images"
    out_labels = Path(output_dir) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print("--- Début de l'augmentation (Luminosité Aléatoire) ---")

    for label_file in Path(label_dir).glob("*.txt"):
        img_path = Path(img_dir) / (label_file.stem + ".jpg")
        if not img_path.exists(): continue

        img = cv2.imread(str(img_path))
        if img is None: continue

        # 1. Copie de l'Original
        shutil.copy(img_path, out_images / img_path.name)
        shutil.copy(label_file, out_labels / label_file.name)

        # 2. Génération d'une version SOMBRE aléatoire
        img_dark = ajuster_luminosite_aleatoire(img, mode="dark")
        dark_name = f"{label_file.stem}_low"
        cv2.imwrite(str(out_images / f"{dark_name}.jpg"), img_dark)
        shutil.copy(label_file, out_labels / f"{dark_name}.txt")

        # 3. Génération d'une version CLAIRE aléatoire
        img_bright = ajuster_luminosite_aleatoire(img, mode="bright")
        bright_name = f"{label_file.stem}_high"
        cv2.imwrite(str(out_images / f"{bright_name}.jpg"), img_bright)
        shutil.copy(label_file, out_labels / f"{bright_name}.txt")

    print(f"Terminé ! Chaque image a maintenant des variantes basse et haute luminosité uniques.")

if __name__ == "__main__":
    IMG_DIR = parent_dir/"Dataset_Basev2"/"train"/"images"
    LABEL_DIR = parent_dir/"Dataset_Basev2"/"train"/"labels"
    OUTPUT_DIR = parent_dir/ "Dataset_Basev2" /"Augmentations"/"Random_HSV"

    if IMG_DIR.exists() and LABEL_DIR.exists():
        augment_luminosite_random(IMG_DIR, LABEL_DIR, OUTPUT_DIR)
        print(f"--- Fin du script : {OUTPUT_DIR} termine avec succes ---")
    else:
        print("ERREUR : Les dossiers sources (images ou labels) sont introuvables.")
        print(f"Cherche dans : {IMG_DIR}")