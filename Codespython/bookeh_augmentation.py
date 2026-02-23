import cv2
import os
import shutil
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
parent_dir = BASE_DIR.parent;
def augment_blur_only(img_dir, label_dir, output_dir):
    # Création des dossiers de sortie
    out_images = Path(output_dir) / "images"
    out_labels = Path(output_dir) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print("--- Début de l'augmentation (Flou Gaussien uniquement) ---")

    for label_file in Path(label_dir).glob("*.txt"):
        img_path = Path(img_dir) / (label_file.stem + ".jpg")
        
        # Vérifier si l'image existe
        if not img_path.exists():
            continue

        # 1. Lecture de l'image
        img = cv2.imread(str(img_path))
        if img is None: continue
        
        # 2. SAUVEGARDE DES ORIGINAUX
        shutil.copy(img_path, out_images / img_path.name)
        shutil.copy(label_file, out_labels / label_file.name)
        
        # 3. CRÉATION DU FLOU GAUSSIEN
        # (5, 5) est la taille du noyau pour un flou léger
        img_blur = cv2.GaussianBlur(img, (5, 5), 0)
        
        # 4. SAUVEGARDE DE LA VERSION FLOU
        blur_name = f"{label_file.stem}_blur"
        
        # Sauvegarde de l'image .jpg
        cv2.imwrite(str(out_images / f"{blur_name}.jpg"), img_blur)
        
        # Sauvegarde du label .txt (copie identique du label original)
        shutil.copy(label_file, out_labels / f"{blur_name}.txt")

    print(f"Opération terminée ! Tes fichiers (originaux + flous) sont ici : {output_dir}")

#EXECUTION
if __name__ == "__main__":
    IMG_DIR = parent_dir/"Dataset_Basev2"/"train"/"images"
    LABEL_DIR = parent_dir/"Dataset_Basev2"/"train"/"labels"
    OUTPUT_DIR = parent_dir/ "Dataset_Basev2" /"Augmentations"/"LowRes"

    if IMG_DIR.exists() and LABEL_DIR.exists():
        augment_dark_noise_dataset(IMG_DIR, LABEL_DIR, OUTPUT_DIR)
        print(f"--- Fin du script : {OUTPUT_DIR} termine avec succes ---")
    else:
        print("ERREUR : Les dossiers sources (images ou labels) sont introuvables.")
        print(f"Cherche dans : {IMG_DIR}"))