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
        #r"P:\Videos\projet\Dataset_Basev2\train\augmentation_combine.py",
        #r"P:\Videos\projet\Dataset_Basev2\train\augmentation_light_hsv_rand.py",
        #r"P:\Videos\projet\Dataset_Basev2\train\augmentation_motion_blur.py",
        r"P:\Videos\projet\Dataset_Basev2\train\augmentation_rotation.py",
        r"P:\Videos\projet\Dataset_Basev2\train\augmentation_zoom.py",
        r"P:\Videos\projet\Dataset_Basev2\train\augmentation_profondeur.py",
        r"P:\Videos\projet\Dataset_Basev2\train\augmentation_distortion.py",
        r"P:\Videos\projet\Dataset_Basev2\train\augmentation_lowres.py",
        "train_Apres_Tuning.py",
        "trainv4.py"
        
    ]
    
    executer_scripts(mes_scripts)
    