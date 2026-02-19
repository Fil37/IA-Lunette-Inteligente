# -*- coding: utf-8 -*-
import subprocess
import os
import sys

# --- DÉTECTION AUTOMATIQUE ---
# Récupère le dossier où se trouve ce script
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# Récupère le chemin du Python actuel (celui de ton venv s'il est activé)
VENV_PYTHON = sys.executable
LOG_FILE = "training_output.log"

# Commande : On lance ton script trainv4.py pour garder ta logique personnalisée
TRAIN_SCRIPT = "trainv4.py"

def launch():
    print(f"🚀 Initialisation du lancement portable...")
    print(f"📂 Dossier détecté : {PROJECT_DIR}")
    print(f"🐍 Python détecté : {VENV_PYTHON}")

    # Vérification de la présence du script d'entraînement
    if not os.path.exists(os.path.join(PROJECT_DIR, TRAIN_SCRIPT)):
        print(f"❌ Erreur : {TRAIN_SCRIPT} est introuvable dans ce dossier !")
        return

    # Construction de la commande nohup
    # On lance 'python trainv4.py' en arrière-plan
    full_command = f"nohup {VENV_PYTHON} {TRAIN_SCRIPT} > {LOG_FILE} 2>&1 &"
    
    try:
        os.chdir(PROJECT_DIR)
        subprocess.Popen(full_command, shell=True)
        
        print("\n✅ Entraînement lancé avec succès !")
        print(f"📈 Tu peux fermer ta session. Logs : {LOG_FILE}")
        print(f"🔍 Commande pour suivre : tail -f {LOG_FILE}")
        
    except Exception as e:
        print(f"❌ Erreur lors du lancement : {e}")

if __name__ == "__main__":
    launch()