from pathlib import Path

def synchroniser_datasets(dossier_txt, dossier_images):
    path_txt = Path(dossier_txt)
    path_img = Path(dossier_images)
    
    extensions_images = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    # 1. On liste tout ce qu'on a
    fichiers_txt = {f.stem for f in path_txt.glob("*.txt")}
    fichiers_img = {f.stem for f in path_img.iterdir() if f.suffix.lower() in extensions_images}

    print(f"État initial : {len(fichiers_img)} images et {len(fichiers_txt)} fichiers texte.")

    # --- ÉTAPE A : Supprimer les IMAGES qui n'ont pas de TEXTE ---
    compteur_img = 0
    for f in path_img.iterdir():
        if f.suffix.lower() in extensions_images and f.stem not in fichiers_txt:
            print(f"Suppression image orpheline : {f.name}")
            f.unlink()
            compteur_img += 1

    # --- ÉTAPE B : Supprimer les TEXTES qui n'ont pas d'IMAGE ---
    compteur_txt = 0
    for f in path_txt.glob("*.txt"):
        if f.stem not in fichiers_img:
            print(f"Suppression texte orphelin : {f.name}")
            f.unlink()
            compteur_txt += 1

    print(f"--- Nettoyage terminé ---")
    print(f"Images supprimées : {compteur_img}")
    print(f"Textes supprimés : {compteur_txt}")

# --- CONFIGURATION ---
DOSSIER_TEXTE = r"P:\Videos\IA-Lunette-Inteligente\Dataset_Basev2\train\labels"
DOSSIER_IMAGES = r"P:\Videos\IA-Lunette-Inteligente\Dataset_Basev2\train\images"

if __name__ == "__main__":
    if Path(DOSSIER_TEXTE).exists() and Path(DOSSIER_IMAGES).exists():
        synchroniser_datasets(DOSSIER_TEXTE, DOSSIER_IMAGES)
    else:
        print("Erreur : Vérifiez les noms de vos dossiers (labels et images).")