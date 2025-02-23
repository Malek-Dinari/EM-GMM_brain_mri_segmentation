import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_preprocess(image_path, desired_resolution=(200, 200)):
    """
    Charge et prétraite l'image :
    - Conversion en niveaux de gris
    - Redimensionnement et normalisation
    - Extraction du cerveau via seuillage Otsu et sélection du plus grand contour
    """
    # Charger l'image en niveaux de gris
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"L'image n'a pas été trouvée : {image_path}")
    
    # Redimensionner l'image
    image_resized = cv2.resize(image, desired_resolution)
    
    # Normaliser l'image (valeurs entre 0 et 1)
    image_norm = image_resized / 255.0

    # Appliquer un seuillage Otsu pour obtenir un masque binaire
    ret, mask = cv2.threshold((image_norm * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Trouver les contours pour extraire la région du cerveau
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Sélectionner le plus grand contour (supposé être le cerveau)
        largest_contour = max(contours, key=cv2.contourArea)
        # Créer un masque vide et dessiner le plus grand contour
        brain_mask = np.zeros_like(mask)
        cv2.drawContours(brain_mask, [largest_contour], -1, 255, thickness=-1)
    else:
        brain_mask = mask

    # Appliquer le masque pour ne conserver que la région du cerveau
    image_brain = cv2.bitwise_and((image_norm * 255).astype(np.uint8), brain_mask)
    
    return image_brain, brain_mask


if __name__ == "__main__":
    image_path = "data/brainmri1.png"
    image_brain, brain_mask = load_and_preprocess(image_path)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image_brain, cmap='gray')
    plt.title("Image avec extraction du cerveau")
    plt.subplot(1, 2, 2)
    plt.imshow(brain_mask, cmap='gray')
    plt.title("Masque du cerveau")
    plt.show()
