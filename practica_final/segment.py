import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def delete_squares(image):
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

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    tophat = cv2.morphologyEx(result, cv2.MORPH_TOPHAT, kernel)

    final_result = cv2.add(result, tophat)

    return final_result


"""def homogene(image, mask):
    black_pixel_ratio = np.sum(image < 10) / image.size

    band_threshold = int(160 + (1 - black_pixel_ratio) * 80)
    band_mask = cv2.inRange(image, band_threshold, 255)
    homogeneous_band = np.where(band_mask > 0, 255, image).astype(np.uint8)
    result = np.where(mask == 255, homogeneous_band, image).astype(np.uint8)
    result = cv2.equalizeHist(result)
    print(black_pixel_ratio, band_threshold)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (60,60))
    tophat = cv2.morphologyEx(result, cv2.MORPH_TOPHAT, kernel, iterations=1)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tophat = cv2.morphologyEx(tophat, cv2.MORPH_TOPHAT, kernel, iterations=20)

    final_result = cv2.add(result, tophat)

    return final_result
"""

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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
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

    sure_fg, mask = delete_squares(gray)

    result = homogene(sure_fg, mask)
    grad_mag = sobel(result)
    grad_mag = binarize_and_fill_holes(grad_mag)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (30, 30))
    tophat_image = cv2.morphologyEx(grad_mag, cv2.MORPH_TOPHAT, kernel,iterations=7)
    grad_mag = cv2.subtract(grad_mag, tophat_image)
    contours, _ = cv2.findContours(grad_mag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(gray)
    min_contour_area = 1000
    large_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]

    if large_contours:
        cv2.drawContours(mask, large_contours, -1, 255, thickness=cv2.FILLED)
        contours_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contours_image, large_contours, -1, (0, 0, 255), thickness=15)

    segmented = cv2.bitwise_and(gray, gray, mask=mask)

    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.title("Imagen Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Segmentación Inicial")
    plt.imshow(result, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Magnitud de sobel")
    plt.imshow(grad_mag, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Magnitud de sobel")
    plt.imshow(contours_image, cmap='gray')
    plt.axis('off')

    plt.show()
    return segmented


base_dir = "Material Mama"
out_dir = "ProcMM"
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.lower().endswith('.jpg'):
            image_path = os.path.join(root, file)
            print(f"Procesando imagen: {image_path}")

            region_inside = segment(image_path)

            output_path = os.path.join(out_dir, os.path.relpath(root, base_dir), f"{file}")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, region_inside)
            print(f"Imagen procesada guardada en: {output_path}")


"""segment("Material Mama/Glandular-denso/mdb033.jpg")
segment("Material Mama/Graso/mdb060.jpg")"""



