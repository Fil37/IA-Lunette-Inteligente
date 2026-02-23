# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import albumentations as A
#CodesPython
base_dir = Path(__file__).resolve().parent

# Dossier Projet
parent_dir = base_dir.parent

def appliquer_dropout(image, bboxes, class_labels):
    """
    Applique des occlusions (rectangles noirs) via Albumentations.
    """
    transform = A.Compose([
        A.CoarseDropout(
            max_holes=8,      # Nombre maximum de rectangles
            max_height=32,    # Hauteur max des rectangles
            max_width=32,     # Largeur max des rectangles
            min_holes=4,      # Nombre minimum de rectangles
            p=1.0
        )
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    return augmented["image"], augmented["bboxes"], augmented["class_labels"]

def augment_dropout_dataset(img_dir, label_dir, output_dir):
    out_images = Path(output_dir) / "images"
    out_labels = Path(output_dir) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print(f"--- Debut de l'augmentation : CoarseDropout -> {output_dir} ---")

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

        # 2. Version avec DROPOUT
        try:
            img_drop, boxes_drop, labels_drop = appliquer_dropout(img, bboxes, class_labels)
            
            # Sauvegarde Image avec suffixe _dropout
            cv2.imwrite(str(out_images / f"{label_file.stem}_dropout.jpg"), img_drop)
            
            # Sauvegarde Labels avec suffixe _dropout
            with open(out_labels / f"{label_file.stem}_dropout.txt", "w") as f:
                for box, cls in zip(boxes_drop, labels_drop):
                    f.write(f"{cls} {' '.join(map(str, box))}\n")
        except Exception as e:
            print(f"Erreur sur {label_file.name} : {e}")

# --- EXECUTION ---
if __name__ == "__main__":
    IMG_DIR = parent_dir/"Dataset_Basev2"/"train"/"images"
    LABEL_DIR = parent_dir/"Dataset_Basev2"/"train"/"labels"
    OUTPUT_DIR = parent_dir/"Dataset_Basev2"/"Augmentations"/"Dropout_Results"

    augment_dropout_dataset(IMG_DIR, LABEL_DIR, OUTPUT_DIR)
    
    if IMG_DIR.exists() and LABEL_DIR.exists():
        augment_dropout_dataset(IMG_DIR, LABEL_DIR, OUTPUT_DIR)
        print(f"--- Fin du script : {OUTPUT_DIR} termine avec succes ---")
    else:
        print("ERREUR : Les dossiers sources (images ou labels) sont introuvables.")
        print(f"Cherche dans : {IMG_DIR}")