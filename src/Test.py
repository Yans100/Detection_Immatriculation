# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor
#
# # Charger l'image
# doc = DocumentFile.from_images("Fichiers/image.jpg")
# model = ocr_predictor(pretrained=True)
# result = model(doc)
#
# # Extraire les mots avec confiance > 0.95
# extracted_words = []
#
# for page in result.pages:
#     for block in page.blocks:
#         for line in block.lines:
#             for word in line.words:
#                 if word.confidence > 0.98:
#                     extracted_words.append(word.value)
#
# # Joindre les mots extraits en une seule chaîne
# final_text = "".join(extracted_words)
# print(final_text)

##########################################################################################################

# from doctr.io import DocumentFile
# from doctr.models import ocr_predictor
# import cv2
#
#
# # Charger l'image avec OpenCV (par défaut en BGR)
# img_bgr = cv2.imread("Fichiers/image.jpg")
#
# # Convertir en RGB, car docTR attend des images RGB
# img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#
# # Charger le modèle OCR
# model = ocr_predictor(pretrained=True)
#
# # Passer l'image sous forme de liste (même pour une seule image)
# result = model([img_rgb])
#
# # Extraire les mots avec confiance > 0.95
# extracted_words = []
#
# for page in result.pages:
#     for block in page.blocks:
#         for line in block.lines:
#             for word in line.words:
#                 if word.confidence > 0.98:
#                     extracted_words.append(word.value)
#
# # Joindre les mots extraits en une seule chaîne
# final_text = "".join(extracted_words)
# print(final_text)

##########################################################################################################

from doctr.models import ocr_predictor
import cv2
from ultralytics import YOLO
import numpy as np
import re
import torch

print("Torch version :", torch.__version__)
print("CUDA disponible :", torch.cuda.is_available())
print("Nom du GPU :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Pas de GPU détecté")
print("Version CUDA :", torch.version.cuda)

# Charger les modèles
modelY = YOLO("yolo11s-seg.pt")
modelY.to('cuda')  # force sur GPU
modelO = ocr_predictor(pretrained=True)
liste_valide = ["ABC123", "M33 VJG"]

# Charger les images
image1 = cv2.imread("PhaseII_SIF1033/Fichiers/image.jpg")
image2 = cv2.imread("PhaseII_SIF1033/Fichiers/image2.jpg")

# Vérifier que les images sont lues correctement
if image1 is None or image2 is None:
    print("Erreur de lecture d'une ou des images")
    exit()

# Fonction pour traiter chaque image
def process_image(image):
    image_width = int(image.shape[1])
    image_height = int(image.shape[0])

    results = modelY(image, conf=0.5, classes=[2, 7], device='cuda')   # Classes 2 (Voitures) and 7 (Camions)
    overlay = np.zeros_like(image)

    if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
        masks = results[0].masks

        for i, mask in enumerate(masks):
            print(f"Mask numéro {i}")
            binary_mask = mask.data.cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
            binary_mask = cv2.resize(binary_mask, (image_width, image_height))

            # Trouver le bounding box du mask pour extraire le véhicule
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue
            x, y, w, h = cv2.boundingRect(contours[0])
            vehicule_roi = image[y:y + h, x:x + w]
            cv2.imshow("roi", vehicule_roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # Convertir la région du véhicule en RGB pour utilisation avec docTR
            plate_RGB = cv2.cvtColor(vehicule_roi, cv2.COLOR_BGR2RGB)
            result = modelO([plate_RGB])

            # Extraire les mots avec confiance > 0.98
            extracted_words = []

            for page in result.pages:
                for block in page.blocks:
                    for line in block.lines:
                        for word in line.words:
                            if word.confidence > 0.99:
                                extracted_words.append(word.value)

            def has_upper_and_digit(text):
                return any(c.isupper() for c in text) and any(c.isdigit() for c in text)

            def is_valid_plate(word):
                return bool(re.fullmatch(r'[A-Z0-9]{7}', word)) and has_upper_and_digit(word)

            def is_valid_plate_blocks(block1, block2):
                if re.fullmatch(r'[A-Z0-9]{3}', block1) and re.fullmatch(r'[A-Z0-9]{3}', block2):
                    combined = block1 + block2
                    return has_upper_and_digit(combined)
                return False

            # Vérification des mots extraits
            for word in extracted_words:
                if is_valid_plate(word):
                    final_text = word
                    break
            else:
                final_text = ""
                for i in range(len(extracted_words) - 1):
                    block1, block2 = extracted_words[i], extracted_words[i + 1]
                    if is_valid_plate_blocks(block1, block2):
                        final_text = f"{block1} {block2}"
                        break

            print("Plaque détectée :", final_text)

            # Appliquer la couleur selon le texte détecté
            if final_text in liste_valide:
                print("Dans liste_valide !!!")
                color = [0, 255, 0]  # Vert
            elif len(final_text) < 6:
                print("< 6")
                color = [0, 165, 255]  # Orange
            else:
                print("Else")
                color = [0, 0, 255]  # Rouge

            # Création du mask coloré
            colored_mask = np.zeros_like(image)
            colored_mask[binary_mask > 0] = color
            overlay = cv2.addWeighted(overlay, 1, colored_mask, 0.6, 0)

        result_image = cv2.addWeighted(image, 1, overlay, 0.5, 0)
        cv2.imshow("result", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Traiter les deux images
process_image(image1)
process_image(image2)



