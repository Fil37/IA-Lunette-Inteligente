# -*- coding: utf-8 -*-
import subprocess
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
parent_dir = BASE_DIR.parent;

def executer_scripts(liste_scripts):
    for script in liste_scripts:
        if not os.path.exists(script):
            print(f" ERREUR : Le fichier est introuvable : {script}")
            continue

        print(f"--- Lancement de : {script}")
        
        # Récupération du dossier où se trouve le script
        dossier_du_script = os.path.dirname(os.path.abspath(script))
        
        try:
            # sys.executable utilise le Python de ton environnement Conda actuel
            resultat = subprocess.run(
                [sys.executable, script], 
                check=True,
                cwd=dossier_du_script
            )
            print(f" {os.path.basename(script)} terminé avec succès\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur lors de l'exécution de {script} : {e}")
            break

if __name__ == "__main__":
    # RACINE du projet (pour simplifier les modifications plus tard)
    train_path = parent_dir / "Dataset_Basev2" / "train"
    code_path = parent_dir / "Codespython"

    mes_scripts = [
        train_path / "augmentation_combine.py",
        code_path / "trainv4.py"
    ]
    
    executer_scripts(mes_scripts)
    