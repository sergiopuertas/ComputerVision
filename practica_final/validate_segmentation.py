import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import os

def xor_and_border_masks(mask1_path, mask2_path):

    mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
    segmented_mask = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)

    if mask1 is None or segmented_mask is None:
        raise ValueError("No se pudieron cargar una o ambas máscaras correctamente.")

    xor_mask = cv2.bitwise_xor(mask1, segmented_mask)
    return segmented_mask, xor_mask


def region_contrast(image, mask):

    region = cv2.bitwise_and(image, image, mask=mask)

    glcm = graycomatrix(region, distances=[1], angles=[0], symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]

    return contrast


def interregion_contrast(image_path, mask1_path, mask2_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("No se pudo cargar la imagen correctamente.")

    segmented_mask, xor_mask = xor_and_border_masks(mask1_path, mask2_path)

    mask_contrast = region_contrast(image, segmented_mask)
    xor_contrast = region_contrast(image, xor_mask)

    contrast_diff = abs(mask_contrast - xor_contrast)

    return mask_contrast, xor_contrast, contrast_diff


def process_all_images(image_folder, proc_first_folder, proc_mm_folder, classes):

    contrast_diff_total = 0
    total_images = 0

    for class_name in classes:
        first_mask_folder = os.path.join(proc_first_folder, class_name)
        mm_mask_folder = os.path.join(proc_mm_folder, class_name)
        image_folder_class = os.path.join(image_folder, class_name)

        for image_filename in os.listdir(image_folder_class):
            image_path = os.path.join(image_folder_class, image_filename)
            first_mask_path = os.path.join(first_mask_folder, image_filename)
            mm_mask_path = os.path.join(mm_mask_folder, image_filename)

            if os.path.isfile(image_path) and os.path.isfile(first_mask_path) and os.path.isfile(mm_mask_path):
                # Calcular el contraste interregión entre la máscara y la región XOR
                mask_contrast, xor_contrast, contrast_diff = interregion_contrast(image_path, first_mask_path, mm_mask_path)

                print(f"Contraste región mamaria en {class_name}/{image_filename}: {mask_contrast:.4f}")
                print(f"Contraste región muscular en {class_name}/{image_filename}: {xor_contrast:.4f}")
                print(f"Diferencia de contraste en {class_name}/{image_filename}: {contrast_diff:.4f}\n")

                contrast_diff_total += contrast_diff
                total_images += 1

    contrast_diff_avg = contrast_diff_total / total_images if total_images > 0 else 0

    return {
        "Diferencia media de contraste": contrast_diff_avg,
        "Total de imágenes procesadas": total_images,
    }


image_folder = "Material Mama"
proc_first_folder = "ProcFirst"
proc_mm_folder = "ProcMM"
classes = ["Glandular-denso", "Glandular-graso", "Graso"]

results = process_all_images(image_folder, proc_first_folder, proc_mm_folder, classes)

print("Resultados:")
for key, value in results.items():
     print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
