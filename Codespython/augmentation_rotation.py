import cv2
import numpy as np
import os
import shutil
import random
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
parent_dir = BASE_DIR.parent;
def rotate_image_and_labels(image, labels, angle):
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    # 1. Matrice de rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 2. Calcul du crop "sans bords noirs"
    abs_angle = abs(np.deg2rad(angle))
    sin_a = np.sin(abs_angle)
    cos_a = np.cos(abs_angle)
    
    # Dimensions du rectangle de contenu utile (inscrit)
    new_w = int(w * cos_a - h * sin_a)
    new_h = int(h * cos_a - w * sin_a)
    new_w, new_h = max(1, abs(new_w)), max(1, abs(new_h))
    
    # Rotation de l'image complète
    rotated = cv2.warpAffine(image, M, (w, h))
    
    # Crop
    x1 = int(center[0] - new_w / 2)
    y1 = int(center[1] - new_h / 2)
    rotated_cropped = rotated[y1:y1+new_h, x1:x1+new_w]
    
    new_labels = []
    for label in labels:
        cls, xc, yc, wb, hb = label
        
        # 3. Calcul des 4 coins de la box originale (en pixels)
        x_min, y_min = (xc - wb/2) * w, (yc - hb/2) * h
        x_max, y_max = (xc + wb/2) * w, (yc + hb/2) * h
        corners = np.array([
            [x_min, y_min], [x_max, y_min], 
            [x_min, y_max], [x_max, y_max]
        ])
        
        # 4. Appliquer la rotation aux coins
        # On ajoute une colonne de 1 pour la multiplication matricielle
        ones = np.ones(shape=(len(corners), 1))
        corners_ones = np.concatenate((corners, ones), axis=1)
        transformed_corners = M.dot(corners_ones.T).T
        
        # 5. Ajuster les coordonnées par rapport au CROP
        transformed_corners[:, 0] -= x1
        transformed_corners[:, 1] -= y1
        
        # 6. Nouveau rectangle englobant (AABB)
        nx_min = np.min(transformed_corners[:, 0])
        ny_min = np.min(transformed_corners[:, 1])
        nx_max = np.max(transformed_corners[:, 0])
        ny_max = np.max(transformed_corners[:, 1])
        
        # 7. CLIPPER les coordonnées pour qu'elles ne dépassent jamais [0, 1]
        nx_min, nx_max = np.clip([nx_min, nx_max], 0, new_w)
        ny_min, ny_max = np.clip([ny_min, ny_max], 0, new_h)
        
        # Calculer les nouvelles valeurs YOLO
        n_xc = ((nx_min + nx_max) / 2) / new_w
        n_yc = ((ny_min + ny_max) / 2) / new_h
        n_wb = (nx_max - nx_min) / new_w
        n_hb = (ny_max - ny_min) / new_h
        
        # On ne garde que si la box a encore une surface
        if n_wb > 0.005 and n_hb > 0.005: 
            new_labels.append(f"{int(cls)} {n_xc:.6f} {n_yc:.6f} {n_wb:.6f} {n_hb:.6f}")
            
    return rotated_cropped, new_labels

def augment_rotation(img_dir, label_dir, output_dir):
    out_images = Path(output_dir) / "images"
    out_labels = Path(output_dir) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    for label_file in Path(label_dir).glob("*.txt"):
        img_path = Path(img_dir) / (label_file.stem + ".jpg")
        if not img_path.exists(): continue

        img = cv2.imread(str(img_path))
        with open(label_file, 'r') as f:
            lines = [list(map(float, line.split())) for line in f.readlines()]

        # Sauvegarde Original
        shutil.copy(img_path, out_images / img_path.name)
        shutil.copy(label_file, out_labels / label_file.name)

        # Création Rotation
        angle = random.uniform(-45, 45)
        img_rot, labels_rot = rotate_image_and_labels(img, lines, angle)
        
        if labels_rot:
            rot_name = f"{label_file.stem}_rot"
            cv2.imwrite(str(out_images / f"{rot_name}.jpg"), img_rot)
            with open(out_labels / f"{rot_name}.txt", 'w') as f_out:
                f_out.write("\n".join(labels_rot))

    print(f"Terminé ! Rotation aléatoire appliquée dans : {output_dir}")

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