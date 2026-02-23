from pathlib import Path

def supprimer_labels_orphelins(dossier_txt, dossier_images):
    path_txt = Path(dossier_txt)
    path_img = Path(dossier_images)

    # 1. On liste TOUTES les images présentes (nom sans extension)
    extensions_images = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.jfif'}
    noms_images = {f.stem for f in path_img.iterdir() if f.suffix.lower() in extensions_images}

    print(f"--- Images trouvees : {len(noms_images)} ---")

    compteur = 0

    # 2. On parcourt les fichiers .txt
    for fichier_txt in path_txt.glob("*.txt"):
        # Si le nom du fichier texte n'est PAS dans la liste des images
        if fichier_txt.stem not in noms_images:
            try:
                print(f"Suppression label orphelin : {fichier_txt.name}")
                fichier_txt.unlink()
                compteur += 1
            except Exception as e:
                print(f"Erreur sur {fichier_txt.name}: {e}")

    print(f"--- Termine ! {compteur} fichiers texte supprimes. ---")

# --- CONFIGURATION ---
DOSSIER_TEXTE = r"P:\Videos\IA-Lunette-Inteligente\Dataset_Basev2\train\labels"
DOSSIER_IMAGES = r"P:\Videos\IA-Lunette-Inteligente\Dataset_Basev2\train\images"

if __name__ == "__main__":
    # Verif simple si les dossiers existent
    if Path(DOSSIER_TEXTE).exists() and Path(DOSSIER_IMAGES).exists():
        supprimer_labels_orphelins(DOSSIER_TEXTE, DOSSIER_IMAGES)
    else:
        print("ERREUR : Verifiez que les dossiers 'labels' et 'images' existent bien ici.")