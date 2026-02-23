import os

# Chemin vers tes labels (ajuste si besoin)
label_path = r"P:\Videos\projet\Dataset_Basev2\train\dataset_augmented_motionblur\labels"

print(f"--- Analyse des fichiers dans : {label_path} ---\n")

count_errors = 0
for filename in os.listdir(label_path):
    if filename.endswith(".txt"):
        file_full_path = os.path.join(label_path, filename)
        with open(file_full_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 0 and len(parts) != 5:
                    print(f"❌ FICHIER : {filename}")
                    print(f"   Ligne {i+1} : '{line.strip()}' ({len(parts)} valeurs au lieu de 5)")
                    print("-" * 30)
                    count_errors += 1

if count_errors == 0:
    print("✅ Aucun problème détecté. Le souci vient peut-être de la manière dont le script lit les fichiers.")
else:
    print(f"\nTerminé ! {count_errors} ligne(s) non conforme(s) trouvée(s).")