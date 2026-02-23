# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import albumentations as A

def appliquer_motion_blur(image, bboxes, class_labels):
    """
    Applique un flou de mouvement via Albumentations.
    """
    transform = A.Compose([
        A.MotionBlur(
            blur_limit=7,  # Taille maximale du noyau de flou
            p=1.0
        )
    ], bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))

    augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    return augmented["image"], augmented["bboxes"], augmented["class_labels"]

def augment_motion_blur_dataset(img_dir, label_dir, output_dir):
    out_images = Path(output_dir) / "images"
    out_labels = Path(output_dir) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print("--- Debut de l'augmentation : Motion Blur (Flou de mouvement) ---")

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

        # 2. Version avec FLOU DE MOUVEMENT
        try:
            img_blur, boxes_blur, labels_blur = appliquer_motion_blur(img, bboxes, class_labels)
            
            # Sauvegarde Image avec suffixe _blur
            cv2.imwrite(str(out_images / f"{label_file.stem}_blur.jpg"), img_blur)
            
            # Sauvegarde Labels avec suffixe _blur
            with open(out_labels / f"{label_file.stem}_blur.txt", "w") as f:
                for box, cls in zip(boxes_blur, labels_blur):
                    f.write(f"{cls} {' '.join(map(str, box))}\n")
        except Exception as e:
            print(f"Erreur sur {label_file.name} : {e}")

    print(f"Termine ! Dataset enrichi disponible dans : {output_dir}")

# --- CONFIGURATION ---
augment_motion_blur_dataset(
    img_dir=r"P:\Videos\projet\Dataset_Basev2\train\images", 
    label_dir=r"P:\Videos\projet\Dataset_Basev2\train\labels", 
    output_dir="dataset_augmented_motionblur"
)
print(f"Termine ! Dataset enrichi disponible dans : {output_dir}")

cv2.destroyAllWindows()
import sys
sys.exit(0)
