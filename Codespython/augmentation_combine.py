# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import shutil
import random
from pathlib import Path
import albumentations as A

# --- CONFIGURATION PORTABLE ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJET_ROOT = SCRIPT_DIR.parent

def fix_path(path_obj):
    """Ajoute le prefixe \\?\ pour contourner la limite MAX_PATH de Windows."""
    path_str = str(path_obj.resolve())
    if os.name == 'nt':  # Si on est sur Windows
        if path_str.startswith("\\\\"): # Chemin reseau (UNC)
            return "\\\\?\\UNC\\" + path_str.lstrip("\\")
        else: # Chemin local (C:, P:, etc.)
            return "\\\\?\\" + path_str
    return path_str

def get_transforms():
    """Definit les transformations complexes avec Albumentations (Version 1.4+ compatible)."""
    return {
        "distorted_crop": A.Compose([
            # 1. On retire shift_limit qui n'existe plus
            A.OpticalDistortion(distort_limit=0.5, p=1.0), 
            # 2. On utilise 'size' au lieu de height/width
            A.RandomResizedCrop(size=(640, 640), scale=(0.5, 1.0), p=1.0) 
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["cls"], clip=True, min_visibility=0.2)),
        
        "blur": A.Compose([
            A.MotionBlur(blur_limit=7, p=1.0)
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["cls"], clip=True, min_visibility=0.1)),
        
        "lowres": A.Compose([
            # 3. On regroupe scale_min et scale_max dans un tuple 'scale'
            A.Downscale(scale=(0.25, 0.5), p=1.0)
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["cls"], clip=True, min_visibility=0.1))
    }

def ajuster_luminosite(image, mode="dark"):
    """Simule des conditions jour/nuit en modifiant le canal V (Valeur)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    facteur = random.uniform(0.3, 0.7) if mode == "dark" else random.uniform(1.3, 1.8)
    v = np.clip(v.astype(np.float64) * facteur, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def process_augmentation():
    IMG_DIR = PROJET_ROOT / "Dataset_Basev2" / "train" / "images"
    LABEL_DIR = PROJET_ROOT / "Dataset_Basev2" / "train" / "labels"
    OUT_DIR = PROJET_ROOT / "Dataset_Basev2" /"Augmentations"/"Dataset_Complet_Augmente"
    
    out_img_path = OUT_DIR / "images"
    out_lab_path = OUT_DIR / "labels"
    
    # Creation des dossiers avec Pathlib (plus robuste)
    out_img_path.mkdir(parents=True, exist_ok=True)
    out_lab_path.mkdir(parents=True, exist_ok=True)

    transforms = get_transforms()
    
    print(f"[START] Lancement de l'augmentation globale...")
    print(f"[PATH] Source : {IMG_DIR}")
    print(f"[PATH] Destination : {OUT_DIR}")

    files = list(LABEL_DIR.glob("*.txt"))
    for i, label_file in enumerate(files):
        if i % 10 == 0: 
            print(f"[PROGRESS] {i}/{len(files)} images traitees...")

        img_path = IMG_DIR / (label_file.stem + ".jpg")
        if not img_path.exists():
            img_path = img_path.with_suffix(".png")
            if not img_path.exists(): continue

        img = cv2.imread(str(img_path))
        if img is None: continue
        
        bboxes, cls_ids = [], []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    bboxes.append(list(map(float, parts[1:])))
                    cls_ids.append(int(parts[0]))

        # 1. Copie Originale (avec fix_path pour les chemins longs)
        dest_img = fix_path(out_img_path / img_path.name)
        dest_lab = fix_path(out_lab_path / label_file.name)
        shutil.copy(str(img_path), dest_img) 
        shutil.copy(str(label_file), dest_lab)

        # 2. Albumentations
        for name, aug in transforms.items():
            try:
                res = aug(image=img, bboxes=bboxes, cls=cls_ids)
                if len(res["bboxes"]) > 0:
                    fname = f"{label_file.stem}_{name}"
                    
                    # Fix pour l'ecriture de l'image
                    img_output = fix_path(out_img_path / f"{fname}.jpg")
                    cv2.imwrite(img_output, res["image"])
                    
                    # Fix pour l'ecriture du label .txt
                    lab_output = fix_path(out_lab_path / f"{fname}.txt")
                    with open(lab_output, "w") as f:
                        for box, c in zip(res["bboxes"], res["cls"]):
                            f.write(f"{c} {' '.join([f'{x:.6f}' for x in box])}\n")
            except Exception as e:
                print(f"[ERROR] Transformation {name} sur {label_file.name} : {e}")

        # 3. Luminosite
        for mode in ["dark", "bright"]:
            img_lumi = ajuster_luminosite(img, mode=mode)
            fname = f"{label_file.stem}_{mode}"
            
            # Fix pour l'image et le label
            cv2.imwrite(fix_path(out_img_path / f"{fname}.jpg"), img_lumi)
            shutil.copy(str(label_file), fix_path(out_lab_path / f"{fname}.txt"))

    print(f"\n[SUCCESS] Termine. Dataset genere dans {OUT_DIR}")

if __name__ == "__main__":
    process_augmentation()