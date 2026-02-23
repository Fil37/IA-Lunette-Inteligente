# -*- coding: utf-8 -*-
from ultralytics import YOLO
import torch
import os
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
parent_dir = BASE_DIR.parent;
def train_model(text, save_dir):
    epochs = 70
    batch = 16

    print("====================================")
    print("?? LANCEMENT DE L'ENTRAINEMENT YOLO")
    print(f"Nombre d'epoques prevues : {epochs}")
    print(f"Batch size              : {batch}")
    print("====================================")

    # Vérification GPU
    print("CUDA disponible :", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU utilise :", torch.cuda.get_device_name(0))
        device = 0
    else:
        print("?? GPU non detecte  entrainement CPU")
        device = "cpu"

    # Chargement du modèle
    model = YOLO(parent_dir/"yolo11m.pt")

    try:
        # Entrainement (YOLO affiche les epoques automatiquement)
        model.train(
            data=parent_dir/"yamlFiles"/"data.yaml",
            epochs=epochs,
            patience=5,
            imgsz=640,
            batch=batch,
            device=device,
            project=parent_dir / "Dataset_Basev2",
            optimizer="AdamW",
            lr0=0.002,
            lrf=0.01,
            cos_lr=True,
            warmup_epochs=3,

            mosaic=0.5,
            mixup=0.0,
            close_mosaic=10,

            workers=8,
            cache=False,
            amp=True,

            name="yolo11m_pedestrian_v3",
            exist_ok=True,
            plots=True   
        )

    except RuntimeError as e:
        print("? Erreur pendant l'entrainement :", e)
        return

    print("====================================")
    print("? Entrainement termine")
    print("====================================")

    # Export ONNX
    model.export(format="onnx")
    
# après l'entraînement

    subprocess.call([parent_dir/"powerbat.bat"], shell=True)

if __name__ == "__main__":
    train_model()
model.fit(x_train, y_train, epochs=10)


