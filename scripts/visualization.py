import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import imageio

def train_em_gmm(image, n_components=3, max_iter=100):
    """
    Entraîne le modèle EM-GMM sur l'image prétraitée.
    
    :param image: Image prétraitée (en niveaux de gris, avec extraction du cerveau)
    :param n_components: Nombre de clusters (par défaut 3)
    :param max_iter: Nombre maximum d'itérations pour EM
    :return: Modèle GMM entraîné
    """
    # Aplatir l'image pour correspondre à l'entrée du GMM
    img_flat = image.flatten().reshape(-1, 1)
    # Créer et entraîner le modèle GMM
    gmm = GaussianMixture(n_components=n_components, max_iter=max_iter, random_state=42)
    gmm.fit(img_flat)
    return gmm

def segment_image(gmm, image, brain_mask):
    """
    Segmente l'image à l'aide du modèle GMM entraîné.
    Applique le masque du cerveau pour exclure les zones non pertinentes.
    
    :param gmm: Modèle GMM entraîné
    :param image: Image prétraitée
    :param brain_mask: Masque binaire du cerveau
    :return: Image segmentée (matrice de labels)
    """
    # Aplatir l'image pour la prédiction
    img_flat = image.flatten().reshape(-1, 1)
    # Prédire le cluster de chaque pixel
    labels = gmm.predict(img_flat)
    seg_image = labels.reshape(image.shape)
    
    # Réassigner les pixels en dehors du cerveau (fond) à un label spécifique, par exemple 0
    seg_image[brain_mask == 0] = 0
    return seg_image


def create_gif(frames, output_path, duration=0.5):
    """
    Crée un GIF à partir d'une liste d'images (frames).
    """
    imageio.mimsave(output_path, frames, duration=duration)

def display_segmentation(original, seg_image):
    """
    Affiche côte à côte l'image originale et l'image segmentée.
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Image Originale")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(seg_image, cmap='viridis')
    plt.title("Image Segmentée")
    plt.axis("off")
    plt.show()