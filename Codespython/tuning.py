# -*- coding: utf-8 -*-
from ultralytics import YOLO
import torch
import os
import subprocess
import time

def run_tuning(yaml_path, project_dir):
    """Lance la recherche automatique d'hyperparametres."""
    print(f"\n?? DEBUT DU TUNING : {yaml_path}")
    
    
    # On utilise le modèle Medium comme dans ton script original
    model = YOLO("yolo11m.pt") 

    # Lancement du tuning (Optimisé pour RTX 3080)
    # Note : Le tuning s'arrête de lui-même après le nombre d'itérations
    model.tune(
        data=yaml_path,
        epochs=30,         # Suffisant pour comparer les augmentations
        iterations=20,     # Environ 15h de travail
        batch=8,           # Pour éviter le Out Of Memory sur la 3080
        imgsz=640,
        device=0,
        optimizer="AdamW",
        project=project_dir,
        name="TUNING_FINAL_SESSION",
        use_ray=False

    )

if __name__ == "__main__":
    # Tes chemins
    base_folder = r"P:\Videos\projet\Resultats_Entrainements_Tuning"
    # Je te suggère de choisir ton meilleur dataset actuel (ex: Light_Random)
    # ou ton dataset de base pour trouver les hyperparamètres idéaux.
    target_yaml = r"P:\Videos\projet\data_v2.yaml" 

    if os.path.exists(target_yaml):
        print("CUDA dispo :", torch.cuda.is_available())
        print("GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Aucun GPU")

        try:
            run_tuning(target_yaml, base_folder)
            print("\n? Tuning termine avec succes.")
        except Exception as e:
            print(f"? Erreur pendant le tuning : {e}")
    else:
        print(f"?? Fichier {target_yaml} introuvable.")

    # --- PARTIE EXTINCTION (Toujours exécutée si le script finit ou crash) ---
    print("\n---------------------------------------------------------")
    print("?? Fin de session. Le PC s'eteindra dans 30 secondes...")
    print("Appuyez sur Ctrl+C pour annuler l'extinction.")
    print("---------------------------------------------------------")
    
    time.sleep(30) # Délai de sécurité pour te laisser annuler si tu es devant l'écran
    
    if os.path.exists("powerbat.bat"):
        subprocess.run(["powerbat.bat"], shell=True)
    else:
        # Si le .bat est perdu, on utilise la commande Windows directe
        print("Fichier .bat introuvable, tentative d'arret direct...")
        os.system("shutdown /s /t 1")