# -*- coding: utf-8 -*-
import subprocess
import sys
import os

def executer_scripts(liste_scripts):
    for script in liste_scripts:
        print(f"--- Lancement de : {script}")
        
        # Récupération du dossier où se trouve le script à lancer
        dossier_du_script = os.path.dirname(os.path.abspath(script))
        
        try:
            # On utilise 'cwd' (Current Working Directory) pour que le script 
            # se lance COMME SI on était dans son propre dossier.
            resultat = subprocess.run(
                [sys.executable, script], 
                check=True,
                cwd=dossier_du_script  # Très important pour les chemins internes de trainv4
            )
            print(f" {script} termine avec succes \n")
        except subprocess.CalledProcessError as e:
            print(f"Erreur lors de l'exécution de {script} : {e}")
            break

if __name__ == "__main__":
    # Liste de tes fichiers avec le chemin complet pour trainv4.py
    # Remplace 'C:/Ton/Chemin/Vers/' par le vrai chemin sur ton disque
    mes_scripts = [
        #"augmentation_fog.py",
        #"augmentation_lowres.py",
        #"augmentation_motion_blur.py",
        #"augmentation_night.py",
        #"augmentation_rain.py",
        "augmentation_distortion.py",
        "augmentation_sunflare.py",
        "trainv4.py" # Utilise 'r' devant le chemin
    ]
    
    executer_scripts(mes_scripts)