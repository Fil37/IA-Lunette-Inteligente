# -*- coding: utf-8 -*-
import cv2
import torch
import subprocess
from ultralytics import YOLO
from pathlib import Path

# --- 1. CONFIGURATION DES CHEMINS ---
BASE_DIR = Path(__file__).resolve().parent
parent_dir = BASE_DIR.parent

# Chemin du modele (adapte selon ton dossier de resultats)
model_path = parent_dir / "runs" / "detect" / "yolo11m_pedestrian_v2" / "weights" / "best.pt"

# Fichiers video
video_input = parent_dir / "videotest.mp4"
video_temp = parent_dir / "video_lourde_temp.mp4"
video_final = parent_dir / "video_resultat_track_PROPRE.mp4"

# --- 2. INITIALISATION ---
# Verification du GPU (CUDA) pour une vitesse maximale
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Utilisation du materiel : {'GPU (NVIDIA)' if device == 0 else 'CPU'}")

model = YOLO(str(model_path))

cap = cv2.VideoCapture(str(video_input))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

# Codec mp4v pour une ecriture rapide sans surcharger le processeur
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(str(video_temp), fourcc, fps, (width, height))

# --- 3. BOUCLE DE TRAITEMENT AVEC TRACKING ---
print(f"Analyse en cours... (Appuyez sur 'q' pour quitter)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # .track() stabilise les boites via le filtre de Kalman interne
    results = model.track(
        source=frame,
        persist=True,          # Garde la memoire des IDs des pietons
        tracker="botsort.yaml", # Meilleur choix pour les cameras mobiles (lunettes)
        conf=0.35,              # Evite les detections incertaines qui vibrent
        iou=0.5,
        imgsz=640,
        device=device,
        verbose=False
    )

    # Dessine les boites et les IDs de suivi
    annotated_frame = results[0].plot()
    
    # Enregistre la frame dans le fichier temporaire lourd
    out.write(annotated_frame)

    # Affichage en direct
    cv2.imshow("Tracking YOLO11 - Appuyez sur Q pour quitter", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Fermeture propre
cap.release()
out.release()
cv2.destroyAllWindows()

# --- 4. COMPRESSION H.264 (OPTION B) ---
print("\n--- Phase de compression (H.264) ---")

# On utilise FFmpeg pour transformer les ~600 Mo en ~50 Mo
cmd_ffmpeg = [
    'ffmpeg', '-y', 
    '-i', str(video_temp),
    '-vcodec', 'libx264',
    '-crf', '23',           # Equilibre qualite/poids standard
    '-preset', 'fast',
    str(video_final)
]

try:
    subprocess.run(cmd_ffmpeg, check=True)
    print(f"Succes ! Video finale : {video_final.name}")
    
    # On supprime le gros fichier de 600 Mo devenu inutile
    if video_temp.exists():
        video_temp.unlink()
        print("Fichier temporaire supprime.")

except Exception as e:
    print(f"Erreur compression : {e}")