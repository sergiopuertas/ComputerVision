import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops


def calculate_contrast(image, mask1, mask2):
    """
    Calcula el contraste entre las regiones segmentadas usando GLCM.

    Parameters:
        image (np.array): Imagen original en escala de grises.
        mask1 (np.array): Máscara 1 (región 1 segmentada).
        mask2 (np.array): Máscara 2 (región 2 segmentada).

    Returns:
        dict: Métricas de contraste entre las regiones segmentadas.
    """
    # Asegurarse de que las máscaras sean binarias
    _, mask1 = cv2.threshold(mask1, 127, 255, cv2.THRESH_BINARY)
    _, mask2 = cv2.threshold(mask2, 127, 255, cv2.THRESH_BINARY)

    # Extraer las regiones segmentadas de la imagen original
    region1 = cv2.bitwise_and(image, image, mask=mask1)
    region2 = cv2.bitwise_and(image, image, mask=mask2)

    # Calcular la diferencia entre las máscaras (máscara 1 vs máscara 2)
    region_diff = cv2.bitwise_xor(region1, region2)
    glcm_diff = graycomatrix(region_diff, [1], [0], symmetric=True, normed=True)
    contrast_diff = graycoprops(glcm_diff, 'contrast')[0, 0]

    return contrast_diff


def process_all_images(image_folder, mask1_folder, mask2_folder):
    """
    Procesa todas las imágenes en una carpeta y calcula el contraste promedio.

    Parameters:
        image_folder (str): Carpeta que contiene las imágenes originales.
        mask1_folder (str): Carpeta que contiene las máscaras 1.
        mask2_folder (str): Carpeta que contiene las máscaras 2.

    Returns:
        dict: Métricas promedio de contraste entre todas las imágenes procesadas.
    """
    contrast_diff_total = 0
    count = 0

    # Iterar sobre las imágenes en la carpeta
    for image_filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_filename)
        mask1_path = os.path.join(mask1_folder, image_filename)
        mask2_path = os.path.join(mask2_folder, image_filename)

        # Comprobar si la imagen y las máscaras existen
        if os.path.exists(image_path) and os.path.exists(mask1_path) and os.path.exists(mask2_path):
            # Cargar imagen y máscaras
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
            mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)

            # Calcular el contraste entre las dos regiones segmentadas
            contrast_diff = calculate_contrast(image, mask1,  mask2)
            print(f"Contraste entre regiones en {image_filename}: {contrast_diff:.4f}")
            contrast_diff_total += contrast_diff
            count += 1

    contrast_diff_avg = contrast_diff_total / count if count > 0 else 0

    return {
        "Average Contrast Mask Difference": contrast_diff_avg
    }


image_folder = "Material Mama/Glandular-denso/"
mask1_folder = "ProcMM/Glandular-denso/mask1/"
mask2_folder = "ProcMM/Glandular-denso/mask2/"

results = process_all_images(image_folder, mask1_folder, mask2_folder)

print("Resultados Promedio de Contraste Intra-Región:")
for key, value in results.items():
    print(f"{key}: {value:.4f}")
