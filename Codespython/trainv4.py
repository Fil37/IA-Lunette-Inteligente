# -*- coding: utf-8 -*-
from ultralytics import YOLO
import torch
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import time
import gc
import logging
from pathlib import Path

# --- CONFIGURATION DES CHEMINS ---
BASE_DIR = Path(__file__).resolve().parent
parent_dir = BASE_DIR.parent

# Configuration du logging
log_file = parent_dir / "Dataset_Basev2" / "suivi_entrainement.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def plot_history_yolo(save_dir):
    save_dir = Path(save_dir)
    csv_path = save_dir / 'results.csv'
    for _ in range(5): 
        if csv_path.exists(): break
        time.sleep(1)
    try:
        data = pd.read_csv(csv_path)
        data.columns = data.columns.str.strip()
        epochs = data['epoch']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.plot(epochs, data['metrics/mAP50(B)'], label='val_mAP50', color='blue', lw=2)
        ax1.set_title('Precision (mAP50)')
        ax1.set_xlabel('Epochs'); ax1.set_ylabel('mAP50')
        ax1.legend(); ax1.grid(True, alpha=0.3)
        ax2.plot(epochs, data['train/box_loss'], label='Train Box Loss', color='orange')
        ax2.plot(epochs, data['val/box_loss'], label='Val Box Loss', color='red')
        ax2.set_title('Perte (Box Loss)')
        ax2.set_xlabel('Epochs'); ax2.set_ylabel('Loss')
        ax2.legend(); ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'custom_metrics_summary.png')
        plt.close()
    except Exception as e:
        logger.error(f"Erreur lecture CSV : {e}")

def train_model(data_yaml_path, project_dir, run_name, unfreeze_at_epoch):
    epochs_total = 200
    batch = 8
    workers = 0 # Important pour la stabilite reseau
    device = 0 if torch.cuda.is_available() else "cpu"
    freeze_layers = 10
    
    logger.info(f"--- START: {run_name} | Device: {device} | Degel a epoque: {unfreeze_at_epoch} ---")
    
    model = YOLO(parent_dir / "yolo11m.pt")

    def unfreeze_callback(trainer):
        if trainer.epoch == unfreeze_at_epoch:
            logger.info(f"INFO: Degel du modele a epoque {trainer.epoch}. Activation de toutes les couches.")
            for param in trainer.model.parameters():
                param.requires_grad = True

    model.add_callback("on_train_epoch_start", unfreeze_callback)

    try:
        results = model.train(
            data=str(data_yaml_path),
            project=str(project_dir),
            name=run_name,
            epochs=epochs_total,
            patience=25,
            imgsz=640,
            batch=batch,
            device=device,
            workers=workers,
            freeze=freeze_layers,
            optimizer="AdamW",
            plots=True,
            exist_ok=True
        )
        
        final_save_dir = Path(results.save_dir)
        time.sleep(2)
        
        logger.info(f"EVALUATION TEST : {run_name}")
        best_model = YOLO(final_save_dir / 'weights' / 'best.pt')
        test_results = best_model.val(data=str(data_yaml_path), split='test')
        logger.info(f"Resultats Test ({run_name}) - mAP50: {test_results.box.map50:.4f}")

        plot_history_yolo(final_save_dir)
        model.export(format="onnx")

    except Exception as e:
        # exc_info=True permet d'enregistrer TOUTE l'erreur dans le .log
        logger.error(f"Erreur lors de l'execution de {run_name} : {e}", exc_info=True)
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
if __name__ == "__main__":
    base_folder = parent_dir / "Dataset_Basev2" / "Resultats_Entrainements"
    base_folder.mkdir(parents=True, exist_ok=True)
    yaml_dir = parent_dir / "yamlFiles"

    experiments = [
        (yaml_dir / "Data_Complet_Augmente.yaml", "Complet_Augmente_F5", 5),
        (yaml_dir / "Data_Complet_Augmente.yaml", "Complet_Augmente_F10", 10),
        (yaml_dir / "data_v2.yaml", "Base_F5", 5)
    ]
    
    termines = 0
    for yaml_path, run_name, unfreeze_epoch in experiments:
        if yaml_path.exists():
            logger.info(f"\n--- Experience {termines + 1}/{len(experiments)} : {run_name} ---")
            train_model(yaml_path, base_folder, run_name, unfreeze_epoch)
            termines += 1
        else:
            logger.error(f"ERREUR : Fichier YAML introuvable : {yaml_path}")

    logger.info(f"Resume : {termines}/{len(experiments)} entrainements effectues.")
    
    if termines > 0:
        logger.info("Fin du script. Fermeture dans 10 secondes...")
        time.sleep(10)
        bat_script = parent_dir / "powerbat.bat"
        if bat_script.exists():
            subprocess.run([str(bat_script)], shell=True)