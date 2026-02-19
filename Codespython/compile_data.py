# -*- coding: utf-8 -*-
import pandas as pd
import glob
import os

def compiler_performances_yolo_complet(dossier_racine, colonne_map50='metrics/mAP50(B)', colonne_map5095='metrics/mAP50-95(B)'):

    # Recherche récursive de tous les fichiers CSV
    pattern = os.path.join(dossier_racine, "**", "*.csv")
    fichiers = glob.glob(pattern, recursive=True)
    
    if not fichiers:
        print(f"Aucun fichier CSV trouve dans l'arborescence de : {dossier_racine}")
        return None

    resultats = []
    print(f"Traitement de {len(fichiers)} fichiers trouves...")

    for fichier in fichiers:
        try:
            # Lire le CSV
            df = pd.read_csv(fichier, skipinitialspace=True)
            # Nettoyer les noms de colonnes (enlever les espaces avant/après)
            df.columns = df.columns.str.strip()
            
            # Vérification que c'est bien un fichier de résultats (doit contenir mAP50)
            if colonne_map50 not in df.columns:
                continue

            # --- SÉLECTION DE LA MEILLEURE EPOCH ---
            # On trouve l'index de la ligne avec le meilleur mAP@50
            idx_meilleur = df[colonne_map50].idxmax()
            
            # On récupère TOUTE la ligne sous forme de Série, puis dictionnaire
            meilleure_ligne = df.loc[idx_meilleur].to_dict()
            
            # --- RECUPERATION DU NOM DE L'EXPERIENCE ---
            nom_dossier_parent = os.path.basename(os.path.dirname(fichier))
            # Si le dossier s'appelle "weights" ou "logs", on remonte d'un cran
            if nom_dossier_parent in ['weights', 'labels', 'train']:
                 path_parts = os.path.normpath(fichier).split(os.sep)
                 # On prend l'avant-avant dernier si possible
                 nom_dossier_parent = path_parts[-3] if len(path_parts) > 2 else nom_dossier_parent

            # --- CONSTRUCTION DU RESULTAT ---
            # On prépare les métadonnées (Nom, Chemin)
            meta_data = {
                'Nom Experience': nom_dossier_parent,
                'Chemin Complet': fichier
            }
            
            # FUSION : On met les métadonnées + toutes les colonnes du CSV (losses, lr, time, etc.)
            # Les ** permettent de fusionner les deux dictionnaires
            ligne_finale = {**meta_data, **meilleure_ligne}
            
            resultats.append(ligne_finale)

        except Exception as e:
            print(f"Erreur sur {fichier}: {e}")

    # Création du DataFrame final
    df_final = pd.DataFrame(resultats)
    
    if not df_final.empty:
        # On trie par mAP@50 (descendant) pour avoir le meilleur modèle en premier
        # On vérifie que la colonne existe bien avant de trier
        if colonne_map50 in df_final.columns:
            df_final = df_final.sort_values(by=colonne_map50, ascending=False)
            
        # --- REORGANISATION DES COLONNES (Optionnel mais propre) ---
        # On met 'Nom Experience' et 'epoch' au début, le reste ensuite
        cols = list(df_final.columns)
        first_cols = ['Nom Experience', 'epoch'] 
        # On garde ceux qui existent dans first_cols, puis on ajoute le reste
        new_order = [c for c in first_cols if c in cols] + [c for c in cols if c not in first_cols]
        df_final = df_final[new_order]
        
    else:
        print("Aucun fichier valide trouves.")
        return None
    
    return df_final

# --- UTILISATION ---

dossier_racine = "./Resultats_Entrainements" # <--- METTRE VOTRE DOSSIER ICI

df_comparatif = compiler_performances_yolo_complet(
    dossier_racine,
    colonne_map50='metrics/mAP50(B)' # Vérifiez que c'est bien le nom exact dans vos CSV
)

if df_comparatif is not None:
    print("\n--- APERU (5 meilleures experiences) ---")
    # Affiche juste les 5 premières lignes dans la console pour vérifier
    pd.set_option('display.max_columns', None) # Pour voir toutes les colonnes dans le print
    print(df_comparatif.head(5))

    # --- EXPORT EXCEL COMPLET ---
    nom_fichier_excel = "comparatif_complet_toutes_metrics_v2.xlsx"
    df_comparatif.to_excel(nom_fichier_excel, index=False)
    
    print(f"\nTermine ! Toutes les donnees (losses, lr, time, metrics...) sont dans : {nom_fichier_excel}")