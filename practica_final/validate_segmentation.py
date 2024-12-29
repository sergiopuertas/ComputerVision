import cv2
import numpy as np
import os
from skimage.feature import graycomatrix, graycoprops


def sliding_window_contrast(image, mask, window_size=(50, 50)):
    """
    Calcula el contraste de la imagen utilizando una ventana deslizante sobre la región extraída por mask1.

    Parameters:
        image (np.array): Imagen original en escala de grises.
        mask (np.array): Máscara .
        window_size (tuple): Tamaño de la ventana deslizante (alto, ancho).

    Returns:
        contrast_values (list): Lista con los valores de contraste calculados en cada ventana.
    """

    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    region = cv2.bitwise_and(image, image, mask=mask2)

    height, width = region.shape
    window_height, window_width = window_size

    contrast_values = []

    for y in range(0, height - window_height + 1, window_height // 2):
        for x in range(0, width - window_width + 1, window_width // 2):
            window = region[y:y + window_height, x:x + window_width]

            if np.count_nonzero(window) > 0:
                glcm = graycomatrix(window, distances=[1], angles=[0], symmetric=True, normed=True)

                contrast = graycoprops(glcm, 'contrast')[0, 0]
                contrast_values.append(contrast)

    return np.mean(contrast_values)


def process_all_images(image_folder, mask_base_folder, classes):
    """
    Procesa todas las imágenes en una carpeta y calcula el contraste promedio.

    Parameters:
        image_folder (str): Carpeta que contiene las imágenes originales.
        mask_base_folder (str): Carpeta base que contiene las máscaras.
        classes (list): Lista de subcarpetas (clases) dentro de la carpeta de máscaras.

    Returns:
        dict: Métricas promedio de contraste entre todas las imágenes procesadas.
    """
    contrast_diff_total = 0
    total_images = 0

    for class_name in classes:
        class_mask_folder = os.path.join(mask_base_folder, class_name)
        image_fold= os.path.join(image_folder, class_name)
        for image_filename in os.listdir(image_fold):
            image_path = os.path.join(image_fold, image_filename)
            mask_path = os.path.join(class_mask_folder, image_filename)

            if (
                os.path.isfile(image_path)
                and os.path.isfile(mask_path)
            ):

                contrast_diff = sliding_window_contrast(image_path, mask_path, window_size=(50, 50))
                print(f"Contraste entre regiones en {class_name}/{image_filename}: {contrast_diff:.4f}")

                contrast_diff_total += contrast_diff
                total_images += 1

    contrast_diff_avg = contrast_diff_total / total_images if total_images > 0 else 0

    return {
        "Average Contrast Mask Difference": contrast_diff_avg,
        "Total Images Processed": total_images,
    }



image_folder = "Material Mama/"
mask_folder = "ProcMM/"
classes = ["Glandular-denso", "Glandular-graso", "Graso"]
results = process_all_images(image_folder, mask_folder, classes)
print("Resultados Promedio de Contraste:")
for key, value in results.items():
    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

