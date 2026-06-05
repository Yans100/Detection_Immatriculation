
# Détection de véhicules et lecture de plaques — SIF1033

Application Python de vision par ordinateur combinant YOLO11 (segmentation) et docTR (OCR) pour détecter des véhicules en temps réel, lire leurs plaques d'immatriculation et colorier le masque selon leur statut (autorisé / inconnu / invalide).

## Fonctionnalités

- Segmentation des véhicules (voitures et camions) par YOLO11 avec masques en temps réel
- OCR des plaques d'immatriculation par docTR sur le ROI extrait du masque YOLO
- Validation des plaques : format québécois (3 lettres + 3 chiffres ou 7 caractères alphanumériques)
- Code couleur sur le masque : vert (plaque autorisée), orange (plaque non lue), rouge (plaque non autorisée)
- Source vidéo configurable : webcam ou fichier vidéo
- Incrustation d'un logo en temps réel avec canal alpha
- Affichage de la date et heure en superposition
- Exécution GPU (CUDA) pour YOLO et docTR

## Technologies

- Python
- YOLO11 (Ultralytics) — segmentation d'instance
- docTR — OCR
- OpenCV
- NumPy
- PyTorch (CUDA)
- pyautogui

## Prérequis

```bash
pip install ultralytics python-doctr[torch] opencv-python numpy pyautogui
```

- GPU NVIDIA avec CUDA recommandé pour les performances temps réel
- Le modèle `yolo11s-seg.pt` doit être présent à la racine

## Lancer le projet

```bash
# Lancer le pipeline vidéo temps réel
python Projet_Phase2.py

# Tester sur des images statiques
python Test.py
```

## Structure

```
Projet_Phase2.py          — script principal (pipeline vidéo)
Fonctions.py              — fonctions utilitaires (source, segmentation, OCR, incrustation)
Test.py                   — tests sur images statiques
```

---

Projet universitaire en équipe — cours SIF1033, UQTR.
