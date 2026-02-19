# -*- coding: utf-8 -*-
from ultralytics import YOLO
import os

def auto_label():
    source_images = r"P:\Videos\projet\pedestrian Traffic Light.v4i.multiclass"
    model_path = r"P:\Videos\projet\Resultats_Entrainements\Augmentation_Zoom\weights\best.pt"

    if not os.path.exists(model_path):
        print("? Modele introuvable.")
        return
    
    model = YOLO(model_path)

    print("--- Lancement des predictions en mode STREAM ---")
    
    # On utilise le generateur stream=True pour ne pas saturer la RAM
    results = model.predict(
        source=source_images,
        save_txt=True,      
        conf=0.25,          
        project=r"P:\Videos\projet\runs\detect",
        name="pre_labelsv2",
        exist_ok=True,
        stream=True,         # <--- TRES IMPORTANT : traite une image a la fois
        vid_stride=1,
        save=True           # On ne sauvegarde pas les images dessinees (economise RAM/Disque)
    )

    # Il faut parcourir le generateur pour que le travail se fasse
    count = 0
    for r in results:
        count += 1
        if count % 100 == 0:
            print(f"Images traitees : {count}")

    print(f"\n? Termine ! {count} images analysees.")

if __name__ == '__main__':
    auto_label()