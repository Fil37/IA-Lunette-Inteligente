# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os
import shutil
import random
from pathlib import Path
import albumentations as A

# --- 1. CONFIGURATION PORTABLE ---
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
    """Definit les transformations complexes (Albumentations 1.4+)."""
    return {
        "distorted": A.Compose([
            A.OpticalDistortion(distort_limit=0.5, p=1.0), 
            A.RandomResizedCrop(size=(640, 640), scale=(0.5, 1.0), p=1.0) 
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["cls"], clip=True, min_visibility=0.2)),
        
        "blur": A.Compose([
            A.MotionBlur(blur_limit=7, p=1.0)
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["cls"], clip=True, min_visibility=0.1)),
        
        "lowres": A.Compose([
            A.Downscale(scale=(0.25, 0.5), p=1.0)
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["cls"], clip=True, min_visibility=0.1))
    }

def ajuster_luminosite(image, mode="dark"):
    """Simule des conditions jour/nuit (canal V de HSV)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    facteur = random.uniform(0.3, 0.7) if mode == "dark" else random.uniform(1.3, 1.8)
    v = np.clip(v.astype(np.float64) * facteur, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

def process_augmentation():
    # Chemins Source
    IMG_DIR = PROJET_ROOT / "Dataset_Basev2" / "train" / "images"
    LABEL_DIR = PROJET_ROOT / "Dataset_Basev2" / "train" / "labels"
    
    # Chemin Destination
    OUT_DIR = PROJET_ROOT / "Dataset_Basev2" / "Augmentations" / "Dataset_Complet_Augmente"
    out_img_path = OUT_DIR / "images"
    out_lab_path = OUT_DIR / "labels"
    
    out_img_path.mkdir(parents=True, exist_ok=True)
    out_lab_path.mkdir(parents=True, exist_ok=True)

    transforms = get_transforms()
    
    print(f"[START] Lancement de l'augmentation combinee...")
    print(f"[PATH] Destination : {OUT_DIR}")

    files = list(LABEL_DIR.glob("*.txt"))
    for i, label_file in enumerate(files):
        if i % 10 == 0: 
            print(f"[PROGRESS] {i}/{len(files)} images traitees...")

        # Trouver l'image correspondante (jpg ou png)
        img_path = IMG_DIR / (label_file.stem + ".jpg")
        if not img_path.exists():
            img_path = img_path.with_suffix(".png")
            if not img_path.exists(): continue

        img = cv2.imread(str(img_path))
        if img is None: continue
        
        # Lecture des Bboxes
        bboxes, cls_ids = [], []
        with open(label_file, "r") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 5:
                    bboxes.append(list(map(float, parts[1:])))
                    cls_ids.append(int(parts[0]))

        # --- 1. COPIE ORIGINALE ---
        shutil.copy(str(img_path), fix_path(out_img_path / img_path.name))
        shutil.copy(str(label_file), fix_path(out_lab_path / label_file.name))

        # --- 2. ALBUMENTATIONS (Distorted, Blur, LowRes) ---
        for name, aug in transforms.items():
            try:
                res = aug(image=img, bboxes=bboxes, cls=cls_ids)
                if len(res["bboxes"]) > 0:
                    fname = f"{label_file.stem}_{name}"
                    cv2.imwrite(fix_path(out_img_path / f"{fname}.jpg"), res["image"])
                    with open(fix_path(out_lab_path / f"{fname}.txt"), "w") as f:
                        for box, c in zip(res["bboxes"], res["cls"]):
                            f.write(f"{c} {' '.join([f'{x:.6f}' for x in box])}\n")
            except Exception as e:
                print(f"[ERROR] Albumentations {name} sur {label_file.name} : {e}")

        # --- 3. LUMINOSITE (Dark & Bright) ---
        for mode in ["dark", "bright"]:
            img_lumi = ajuster_luminosite(img, mode=mode)
            fname = f"{label_file.stem}_{mode}"
            cv2.imwrite(fix_path(out_img_path / f"{fname}.jpg"), img_lumi)
            shutil.copy(str(label_file), fix_path(out_lab_path / f"{fname}.txt"))

        # --- 4. ZOOM NATUREL (Sur la premiere box) ---
        if bboxes:
            try:
                h_o, w_o = img.shape[:2]
                xc, yc, wb, hb = bboxes[0]
                # Conversion YOLO -> Pixels
                l = max(0, int((xc - wb/2) * w_o))
                t = max(0, int((yc - hb/2) * h_o))
                r = min(w_o, int((xc + wb/2) * w_o))
                b = min(h_o, int((yc + hb/2) * h_o))
                
                crop = img[t:b, l:r]
                if crop.size > 0:
                    fname = f"{label_file.stem}_zoom"
                    cv2.imwrite(fix_path(out_img_path / f"{fname}.jpg"), crop)
                    # Label : l'objet occupe maintenant tout le crop (centre 0.5)
                    with open(fix_path(out_lab_path / f"{fname}.txt"), "w") as f:
                        f.write(f"{cls_ids[0]} 0.500000 0.500000 1.000000 1.000000\n")
            except Exception as e:
                print(f"[ERROR] Zoom sur {label_file.name} : {e}")

    print(f"\n[SUCCESS] Termine. Dataset genere dans {OUT_DIR}")

if __name__ == "__main__":
    process_augmentation()