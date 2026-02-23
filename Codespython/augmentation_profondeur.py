import cv2
import numpy as np
import os
import shutil
import random
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
parent_dir = BASE_DIR.parent;
def perspective_warp_and_crop(image, labels, magnitude=0.15):
    h, w = image.shape[:2]
    
    # Points d'origine (4 coins)
    src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    
    # 1. Génération des points de destination déformés
    p1 = [random.uniform(0, w*magnitude), random.uniform(0, h*magnitude)]
    p2 = [w - random.uniform(0, w*magnitude), random.uniform(0, h*magnitude)]
    p3 = [w - random.uniform(0, w*magnitude), h - random.uniform(0, h*magnitude)]
    p4 = [random.uniform(0, w*magnitude), h - random.uniform(0, h*magnitude)]
    dst_pts = np.float32([p1, p2, p3, p4])
    
    # 2. Calcul du rectangle de crop (pour éviter le noir)
    # On prend le max des coins gauches/hauts et le min des coins droits/bas
    crop_x1 = int(max(p1[0], p4[0]))
    crop_y1 = int(max(p1[1], p2[1]))
    crop_x2 = int(min(p2[0], p3[0]))
    crop_y2 = int(min(p3[1], p4[1]))
    
    new_w = crop_x2 - crop_x1
    new_h = crop_y2 - crop_y1

    # 3. Matrice de transformation et application
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped_img = cv2.warpPerspective(image, M, (w, h))
    
    # On applique le crop immédiat
    warped_cropped = warped_img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # 4. Transformation des labels
    new_labels = []
    for label in labels:
        cls, xc, yc, wb, hb = label
        
        # Coins de la box originale
        x1_b, y1_b = (xc - wb/2) * w, (yc - hb/2) * h
        x2_b, y2_b = (xc + wb/2) * w, (yc + hb/2) * h
        
        pts = np.array([[[x1_b, y1_b]], [[x2_b, y1_b]], [[x2_b, y2_b]], [[x1_b, y2_b]]], dtype=np.float32)
        transformed_pts = cv2.perspectiveTransform(pts, M)
        
        # Recalcul de l'AABB (Axis Aligned Bounding Box)
        all_x = transformed_pts[:, 0, 0]
        all_y = transformed_pts[:, 0, 1]
        
        # Ajustement par rapport au décalage du crop
        final_x1, final_x2 = np.min(all_x) - crop_x1, np.max(all_x) - crop_x1
        final_y1, final_y2 = np.min(all_y) - crop_y1, np.max(all_y) - crop_y1
        
        # Normalisation par rapport à la taille de l'image découpée
        n_xc = ((final_x1 + final_x2) / 2) / new_w
        n_yc = ((final_y1 + final_y2) / 2) / new_h
        n_wb = (final_x2 - final_x1) / new_w
        n_hb = (final_y2 - final_y1) / new_h
        
        # On garde l'objet s'il est au moins partiellement dans le crop
        if 0 < n_xc < 1 and 0 < n_yc < 1:
            # Sécurité : on borne entre 0 et 1
            n_xc, n_yc = np.clip([n_xc, n_yc], 0, 1)
            n_wb, n_hb = np.clip([n_wb, n_hb], 0, 1)
            new_labels.append(f"{int(cls)} {n_xc:.6f} {n_yc:.6f} {n_wb:.6f} {n_hb:.6f}")
            
    return warped_cropped, new_labels

def augment_perspective_cropped(img_dir, label_dir, output_dir):
    out_images = Path(output_dir) / "images"
    out_labels = Path(output_dir) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print("--- Début de l'augmentation : Perspective + Auto-Crop ---")

    for label_file in Path(label_dir).glob("*.txt"):
        img_path = Path(img_dir) / (label_file.stem + ".jpg")
        if not img_path.exists(): continue

        img = cv2.imread(str(img_path))
        with open(label_file, 'r') as f:
            lines = [list(map(float, line.split())) for line in f.readlines()]

        # Sauvegarde Original
        shutil.copy(img_path, out_images / img_path.name)
        shutil.copy(label_file, out_labels / label_file.name)

        # Création Perspective Cropped
        img_p, labels_p = perspective_warp_and_crop(img, lines)
        
        if labels_p and img_p.size > 0:
            p_name = f"{label_file.stem}_perspective"
            cv2.imwrite(str(out_images / f"{p_name}.jpg"), img_p)
            with open(out_labels / f"{p_name}.txt", 'w') as f_out:
                f_out.write("\n".join(labels_p))

    print(f"Terminé ! Les images sont déformées et recadrées dans : {output_dir}")
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
        print(f"Cherche dans : {IMG_DIR}")