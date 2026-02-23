# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import albumentations as A
BASE_DIR = Path(__file__).resolve().parent
parent_dir = BASE_DIR.parent;
def appliquer_distortion(image, bboxes, class_labels):
    """
    Applique une distorsion optique via Albumentations.
    """
    transform = A.Compose([
        A.OpticalDistortion(
            distort_limit=0.5, 
            shift_limit=0.2, 
            p=1.0
        )
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    return augmented["image"], augmented["bboxes"], augmented["class_labels"]

def augment_distortion_dataset(img_dir, label_dir, output_dir):
    out_images = Path(output_dir) / "images"
    out_labels = Path(output_dir) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print(f"--- Debut de l'augmentation : Distorsion Optique -> {output_dir} ---")

    for label_file in Path(label_dir).glob("*.txt"):
        img_path = Path(img_dir) / (label_file.stem + ".jpg")
        
        if not img_path.exists():
            img_path = img_path.with_suffix(".png")
            if not img_path.exists(): continue

        img = cv2.imread(str(img_path))
        if img is None: continue

        # 1. Chargement des Bboxes (Format YOLO)
        bboxes = []
        class_labels = []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    c, x, y, w, h = map(float, parts)
                    bboxes.append([x, y, w, h])
                    class_labels.append(int(c))

        # 2. Copie de l'Original
        shutil.copy(img_path, out_images / img_path.name)
        shutil.copy(label_file, out_labels / label_file.name)

        # 3. Creation de la version DISTORDUE
        try:
            img_dist, boxes_dist, labels_dist = appliquer_distortion(img, bboxes, class_labels)
            
            new_img_name = f"{label_file.stem}_distorted.jpg"
            cv2.imwrite(str(out_images / new_img_name), img_dist)
            
            new_label_name = f"{label_file.stem}_distorted.txt"
            with open(out_labels / new_label_name, "w") as f:
                for box, cls in zip(boxes_dist, labels_dist):
                    f.write(f"{cls} {' '.join(map(str, box))}\n")
                    
        except Exception as e:
            print(f"Erreur lors du traitement de {img_path.name} : {e}")

# --- EXECUTION ---
if __name__ == "__main__":
    IMG_DIR = parent_dir/"Dataset_Basev2"/"train"/"images"
    LABEL_DIR = parent_dir/"Dataset_Basev2"/"train"/"labels"
    OUTPUT_DIR = parent_dir/"Augmentations"/"Distorted"

    if IMG_DIR.exists() and LABEL_DIR.exists():
        augment_distortion_dataset(IMG_DIR, LABEL_DIR, OUTPUT_DIR)
        print(f"--- Fin du script : {OUTPUT_DIR} termine avec succes ---")
    else:
        print("ERREUR : Les dossiers sources (images ou labels) sont introuvables.")
        print(f"Cherche dans : {IMG_DIR}")
    