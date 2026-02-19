import yaml
import os

# Configuration de base
base_path = r"P:\Videos\projet\Dataset_Basev2"
nc = 6
names = ['greenlight', 'greenlight_car', 'redlight', 'redlight_car', 'yellowlight_car', 'zebracrossing']

# Liste des augmentations et leurs dossiers correspondants
augmentations = {
    "fog": "dataset_augmented_fog",
    "lowres": "dataset_augmented_lowres",
    "motion_blur": "dataset_augmented_motionblur",
    "night": "dataset_augmented_darknoise",
    "noise": "dataset_augmented_noise",
    "rain": "dataset_augmented_rain",
    "cutout": "dataset_augmented_dropout",
    "shadow": "dataset_augmented_shadow",
    "distortion": "dataset_augmented_distorted",
    "sunflare": "dataset_augmented_sunflare"
}

def generate_yamls():
    for key, folder in augmentations.items():
        data = {
            'train': os.path.join(base_path, 'train', folder, 'images'),
            'val': os.path.join(base_path, 'valid', 'images'),
            'test': os.path.join(base_path, 'test', 'images'),
            'nc': nc,
            'names': names
        }
        
        filename = f"data_{key}.yaml"
        with open(filename, 'w') as f:
            # On utilise default_flow_style=False pour avoir un format propre
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        print(f"Fichier créé : {filename}")

if __name__ == "__main__":
    generate_yamls()