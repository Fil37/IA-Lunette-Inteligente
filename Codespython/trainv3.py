# -*- coding: utf-8 -*-
from ultralytics import YOLO
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import subprocess

def plot_history_yolo(save_dir):
    """Lit le fichier results.csv et genere des graphiques personnalises."""
    csv_path = os.path.join(save_dir, 'results.csv')
    if not os.path.exists(csv_path):
        print(f"?? Fichier de resultats introuvable : {csv_path}")
        return

    data = pd.read_csv(csv_path)
    data.columns = data.columns.str.strip()
    epochs = data['epoch']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Précision
    ax1.plot(epochs, data['metrics/mAP50(B)'], label='val_mAP50', color='blue', lw=2)
    ax1.set_title('Precision (mAP50)')
    ax1.set_xlabel('Epochs'); ax1.set_ylabel('mAP50')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # Perte
    ax2.plot(epochs, data['train/box_loss'], label='Train Box Loss', color='orange')
    ax2.plot(epochs, data['val/box_loss'], label='Val Box Loss', color='red')
    ax2.set_title('Perte (Box Loss)')
    ax2.set_xlabel('Epochs'); ax2.set_ylabel('Loss')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'custom_metrics_summary.png'))
    plt.close()

def train_model(data_yaml_path, project_dir, run_name):
    """Entraine YOLO et valide sur le set de TEST final."""
    epochs = 200
    batch = 8
    workers = 0
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"\n?? START: {run_name} | Device: {device}")
    
    model = YOLO("yolo11m.pt")

    try:
        # 1. ENTRAÎNEMENT (Utilise Train et Val)
        results = model.train(
            data=data_yaml_path,
            project=project_dir,
            name=run_name,
            epochs=epochs,
            patience=5,
            imgsz=640,
            batch=batch,
            device=device,
            workers = workers,
            optimizer="AdamW",
            plots=True,
            exist_ok=True
        )
        
        final_save_dir = results.save_dir
        
        # 2. VÉRIFICATION SUR LE TEST (L'examen final)
        print(f"\n?? EVALUATION SUR LE SPLIT TEST : {run_name}")
        # On charge le meilleur poids obtenu
        best_model = YOLO(os.path.join(final_save_dir, 'weights', 'best.pt'))
        
        # split='test' force YOLO à utiliser les images définies dans la ligne 'test:' du YAML
        test_results = best_model.val(data=data_yaml_path, split='test', project=project_dir, name=f"{run_name}_TEST_FINAL")
        
        print(f"? Resultats Test ({run_name}) - mAP50: {test_results.box.map50:.4f}")

        # 3. GRAPHICS & EXPORT
        plot_history_yolo(final_save_dir)
        model.export(format="onnx")

    except Exception as e:
        print(f"? Erreur lors de l'execution de {run_name} : {e}")

if __name__ == "__main__":
    base_folder = r"P:\Videos\projet\Resultats_Entrainements"
    yaml_dir = r"P:\Videos\projet"

    experiments = [

        #(os.path.join(yaml_dir, "data_v2.yaml"), "BaseLine_v2"),
        #(os.path.join(yaml_dir, "data_v2_perspective_zoom.yaml"), "Augmentation_Perspective_Zoom"),
        #(os.path.join(yaml_dir, "data_v2_rotation.yaml"), "Augmentation_Rotation_v2"),
        #(os.path.join(yaml_dir, "data_v2_hsv_rand.yaml"), "Augmentation_Light_Random_v2"),
        #(os.path.join(yaml_dir, "data_v2_hsv.yaml"), "Augmentation_Light_Random_v2"),
        #(os.path.join(yaml_dir, "data_v2_perspective.yaml"), "Augmentation_Perspective_v2"),
        #os.path.join(yaml_dir, "data_v2_blur.yaml"), "Augmentation_Blur_v2")
        (os.path.join(yaml_dir, "data_v2_zoom.yaml"), "Augmentation_Zoom_v2")
        
    ]

    entrainements_termines = 0
    total_prevu = len(experiments)

    # --- BOUCLE D'ENTRAINEMENT ---
    for yaml_path, run_name in experiments:
        if os.path.exists(yaml_path):
            print(f"\n--- Debut de l'experience {entrainements_termines + 1}/{total_prevu} : {run_name} ---")
            
            # Cette fonction est BLOQUANTE : Python attend la fin du train avant de continuer
            train_model(yaml_path, base_folder, run_name)
            
            entrainements_termines += 1
        else:
            print(f"? ERREUR : Le fichier {yaml_path} est introuvable !")

    # --- CONDITION DE FIN ---
    # On vérifie si on a bien parcouru toute la liste
    print(f"\n?? Resume : {entrainements_termines}/{total_prevu} entrainements effectues.")

    if entrainements_termines > 0:
        print("? Tous les modeles ont ete traites. Lancement du script de fin dans 10 secondes...")
        # Optionnel : un petit délai pour te laisser lire la console avant la fermeture
        import time
        time.sleep(10) 
        
        subprocess.run(["powerbat.bat"], shell=True)
    else:
        print("?? Aucun entrainement n'a ete lance (fichiers YAML introuvables). Le script de fin est annule.")