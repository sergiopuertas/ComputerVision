import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def delete_bg(image):
    _, sure_fg = cv2.threshold(image, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    region_inside = cv2.bitwise_and(image, image, mask=mask)

    return region_inside, mask


def homogene(image, mask):
    band_threshold = 187

    band_mask = cv2.inRange(image, band_threshold, 255)

    homogeneous_band = np.where(band_mask > 0, 255, image).astype(np.uint8)
    result = np.where(mask == 255, homogeneous_band, image).astype(np.uint8)
    result = cv2.equalizeHist(result)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    tophat = cv2.morphologyEx(result, cv2.MORPH_TOPHAT, kernel, iterations=3)

    final_result = cv2.add(result, tophat)
    return final_result

def sobel(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_mag = np.uint8(np.clip(grad_x, 0, 255))
    _, grad_mag = cv2.threshold(grad_mag, 0, 255, cv2.THRESH_BINARY)

    grad_y = cv2.Sobel(grad_mag, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.uint8(np.clip(grad_y, 0, 255))
    _, grad_mag = cv2.threshold(grad_mag, 0, 255, cv2.THRESH_BINARY)
    return grad_mag


def binarize_and_fill_holes(sobel_image):
    _, binary_image = cv2.threshold(sobel_image, 10, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 20))
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    inverted = cv2.bitwise_not(closed_image)

    h, w = inverted.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    filled = inverted.copy()
    for x, y in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        cv2.floodFill(filled, mask, (x, y), 0)

    result = cv2.bitwise_or(closed_image, filled)
    return result


def segment(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sure_fg, mask1 = delete_bg(gray)

    result = homogene(sure_fg, mask1)
    sobelim = sobel(result)

    grad_mag = binarize_and_fill_holes(sobelim)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (25, 25))
    tophat_image = cv2.morphologyEx(grad_mag, cv2.MORPH_TOPHAT, kernel,iterations=3)
    grad_mag = cv2.subtract(grad_mag, tophat_image)
    contours, _ = cv2.findContours(grad_mag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask2 = np.zeros_like(gray)
    min_contour_area = 100000
    large_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

    if large_contours:
        cv2.drawContours(mask2, large_contours, -1, 255, thickness=cv2.FILLED)
        contours_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contours_image, large_contours, -1, (0, 0, 255), thickness=15)

    segmented = cv2.bitwise_and(gray, gray, mask=mask2)


    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.title("Imagen Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Procesado inicial")
    plt.imshow(result, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Gradientes de Sobel")
    plt.imshow(sobelim, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Imagen final")
    plt.imshow(contours_image, cmap='gray')
    plt.axis('off')

    plt.show()
    return segmented, mask2


base_dir = "Material Mama"
out_dir = "ProcMM"

for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith('.jpg'):
            image_path = os.path.join(root, file)
            print(f"Procesando imagen: {image_path}")

            region_inside, mask = segment(image_path)

            class_dir = os.path.relpath(root, base_dir)
            output_path_mask = os.path.join(out_dir, class_dir, file)
            os.makedirs(os.path.dirname(output_path_mask), exist_ok=True)
            cv2.imwrite(output_path_mask, mask)
            print(f"Máscara guardada en: {output_path_mask}")



