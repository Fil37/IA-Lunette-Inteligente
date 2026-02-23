import cv2
import os
import shutil
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
parent_dir = BASE_DIR.parent;
def augment_multi_zoom_naturel(img_dir, label_dir, output_dir):
    # Création des dossiers de sortie
    out_images = Path(output_dir) / "images"
    out_labels = Path(output_dir) / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print("--- Début de l'augmentation (Crops naturels) ---")

    for label_file in Path(label_dir).glob("*.txt"):
        img_path = Path(img_dir) / (label_file.stem + ".jpg")
        if not img_path.exists():
            continue

        # 1. Lecture de l'image et des labels originaux
        img = cv2.imread(str(img_path))
        if img is None: continue
        h_orig, w_orig, _ = img.shape
        
        with open(label_file, 'r') as f:
            lines = f.readlines()
        if not lines: continue

        # 2. Sauvegarde de la copie originale
        shutil.copy(img_path, out_images / img_path.name)
        shutil.copy(label_file, out_labels / label_file.name)

        # 3. Création d'un zoom pour CHAQUE box
        for i, target_line in enumerate(lines):
            data = list(map(float, target_line.split()))
            cls_t, x_ct, y_ct, w_bt, h_bt = data

            # Calcul de la zone de zoom (Côtés x 2)
            zoom_w, zoom_h = w_bt * 2, h_bt * 2

            # Coordonnées de la découpe en pixels
            left = max(0, int((x_ct - zoom_w / 2) * w_orig))
            top = max(0, int((y_ct - zoom_h / 2) * h_orig))
            right = min(w_orig, int((x_ct + zoom_w / 2) * w_orig))
            bottom = min(h_orig, int((y_ct + zoom_h / 2) * h_orig))

            # Découpe de l'image (sans resize)
            img_crop = img[top:bottom, left:right]
            if img_crop.size == 0: continue
            
            h_crop, w_crop, _ = img_crop.shape

            # 4. Recalcul des labels pour cette nouvelle image
            new_labels = []
            for line in lines:
                c, xc, yc, wb, hb = map(float, line.split())
                
                # Conversion en pixels (repère image origine)
                px_c, py_c = xc * w_orig, yc * h_orig
                px_w, px_h = wb * w_orig, hb * h_orig

                # Vérifier si le centre de l'objet est dans la zone découpée
                if left < px_c < right and top < py_c < bottom:
                    # Nouvelles coordonnées relatives à la taille de la découpe
                    nx = (px_c - left) / w_crop
                    ny = (py_c - top) / h_crop
                    nw = px_w / w_crop
                    nh = px_h / h_crop
                    
                    label_str = f"{int(c)} {min(nx,1.):.6f} {min(ny,1.):.6f} {min(nw,1.):.6f} {min(nh,1.):.6f}"
                    new_labels.append(label_str)

            # 5. Sauvegarde
            zoom_name = f"{label_file.stem}_zoom_{i}"
            cv2.imwrite(str(out_images / f"{zoom_name}.jpg"), img_crop)

            with open(out_labels / f"{zoom_name}.txt", 'w') as f_out:
                f_out.write("\n".join(new_labels))

    print(f"Opération terminée ! Les images zoomées ont leurs tailles d'origine (x2).")

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