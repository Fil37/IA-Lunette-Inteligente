import os
import shutil
import random

# --- MODIFICATION DU CHEMIN ---
# On descend dans l'arborescence pour trouver les vraies images
dataset_folder = os.path.join("Dataset", "train", "images")

sample_folder = "sample_100"
sample_size = 100

os.makedirs(sample_folder, exist_ok=True)

all_images = [f for f in os.listdir(dataset_folder)
if os.path.isfile(os.path.join(dataset_folder, f))
and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

# --- AJOUT POUR LA REPRESENTATIVITE (accents retires) ---
# On melange la liste pour casser l'ordre alphabetique ou temporel
random.shuffle(all_images)

if len(all_images) < sample_size:
    raise ValueError(f"Le dataset contient seulement {len(all_images)} images.")

# Tirer l'echantillon (se base maintenant sur la liste melangee)
sample_images = random.sample(all_images, sample_size)

for img in sample_images:
    # --- LES CHEMINS UTILISENT MAINTENANT LE dataset_folder CORRIGE ---
    shutil.copy2(os.path.join(dataset_folder, img), os.path.join(sample_folder, img))

print(f"Echantillon de {sample_size} images cree dans '{sample_folder}' !")