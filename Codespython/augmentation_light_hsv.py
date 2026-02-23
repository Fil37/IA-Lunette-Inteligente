import cv2
import numpy as np
import os
import shutil
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
parent_dir = BASE_DIR.parent;
def ajuster_luminosite(image, facteur):
    # Conversion vers l'espace HSV pour isoler la luminosité (V)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Application du facteur sur le canal V
    # On utilise float64 pour éviter les dépassements avant le découpage (clip)
    v = np.array(v, dtype=np.float64)
    v = v * facteur
    
    # On bloque les valeurs entre 0 et 255 et on repasse en entier 8 bits
    v = np.clip(v, 0, 255).astype(np.uint8)

    # Recomposition de l'image
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def augment_luminosite(img_dir, label_dir, output_dir):
    out_images = Path(output_dir) / "images"
    out_labels = Path(output_dir) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print("--- Début de l'augmentation de luminosité ---")

    for label_file in Path(label_dir).glob("*.txt"):
        img_path = Path(img_dir) / (label_file.stem + ".jpg")
        if not img_path.exists(): continue

        img = cv2.imread(str(img_path))
        if img is None: continue

        # 1. Copie de l'Original
        shutil.copy(img_path, out_images / img_path.name)
        shutil.copy(label_file, out_labels / label_file.name)

        # 2. Version SOMBRE (x 0.6)
        img_dark = ajuster_luminosite(img, 0.6)
        cv2.imwrite(str(out_images / f"{label_file.stem}_dark.jpg"), img_dark)
        shutil.copy(label_file, out_labels / f"{label_file.stem}_dark.txt")

        # 3. Version CLAIRE (x 1.4)
        img_bright = ajuster_luminosite(img, 1.4)
        cv2.imwrite(str(out_images / f"{label_file.stem}_bright.jpg"), img_bright)
        shutil.copy(label_file, out_labels / f"{label_file.stem}_bright.txt")

    print(f"Terminé ! Dataset enrichi disponible dans : {output_dir}")

if __name__ == "__main__":
    IMG_DIR = parent_dir/"Dataset_Basev2"/"train"/"images"
    LABEL_DIR = parent_dir/"Dataset_Basev2"/"train"/"labels"
    OUTPUT_DIR = parent_dir/"Augmentations"/"HSV"

    if IMG_DIR.exists() and LABEL_DIR.exists():
        augment_luminosite(IMG_DIR, LABEL_DIR, OUTPUT_DIR)
        print(f"--- Fin du script : {OUTPUT_DIR} termine avec succes ---")
    else:
        print("ERREUR : Les dossiers sources (images ou labels) sont introuvables.")
        print(f"Cherche dans : {IMG_DIR}")