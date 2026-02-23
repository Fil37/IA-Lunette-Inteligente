# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import albumentations as A
BASE_DIR = Path(__file__).resolve().parent
parent_dir = BASE_DIR.parent;
def appliquer_downscale(image, bboxes, class_labels):
    """
    Applique une reduction de resolution via Albumentations.
    """
    transform = A.Compose([
        A.Downscale(
            scale_min=0.25, # Reduction a 25% de la taille originale
            scale_max=0.5,  # Reduction a 50% de la taille originale
            p=1.0
        )
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    return augmented["image"], augmented["bboxes"], augmented["class_labels"]

def augment_downscale_dataset(img_dir, label_dir, output_dir):
    out_images = Path(output_dir) / "images"
    out_labels = Path(output_dir) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print("--- Debut de l'augmentation : Downscale (Basse Resolution) ---")

    for label_file in Path(label_dir).glob("*.txt"):
        img_path = Path(img_dir) / (label_file.stem + ".jpg")
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Lecture des labels YOLO
        bboxes = []
        class_labels = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    c, x, y, w, h = map(float, parts)
                    bboxes.append([x, y, w, h])
                    class_labels.append(int(c))

        # 1. Copie de l'Original
        shutil.copy(img_path, out_images / img_path.name)
        shutil.copy(label_file, out_labels / label_file.name)

        # 2. Version BASSE RESOLUTION
        try:
            img_low, boxes_low, labels_low = appliquer_downscale(img, bboxes, class_labels)
            
            # Sauvegarde Image avec suffixe _lowres
            cv2.imwrite(str(out_images / f"{label_file.stem}_lowres.jpg"), img_low)
            
            # Sauvegarde Labels avec suffixe _lowres
            with open(out_labels / f"{label_file.stem}_lowres.txt", "w") as f:
                for box, cls in zip(boxes_low, labels_low):
                    f.write(f"{cls} {' '.join(map(str, box))}\n")
        except Exception as e:
            print(f"Erreur sur {label_file.name} : {e}")

    print(f"Termine ! Dataset enrichi disponible dans : {output_dir}")

if __name__ == "__main__":
    IMG_DIR = parent_dir/"Dataset_Basev2"/"train"/"images"
    LABEL_DIR = parent_dir/"Dataset_Basev2"/"train"/"labels"
    OUTPUT_DIR = parent_dir/ "Dataset_Basev2" /"Augmentations"/"LowRes"

    if IMG_DIR.exists() and LABEL_DIR.exists():
        augment_downscale_dataset(IMG_DIR, LABEL_DIR, OUTPUT_DIR)
        print(f"--- Fin du script : {OUTPUT_DIR} termine avec succes ---")
    else:
        print("ERREUR : Les dossiers sources (images ou labels) sont introuvables.")
        print(f"Cherche dans : {IMG_DIR}")