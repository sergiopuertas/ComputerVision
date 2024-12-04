import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import active_contour
from skimage import img_as_float
from skimage.draw import circle_perimeter

def process_with_watershed(image_path):
    # Leer la imagen original
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Aplicar operaciones morfológicas iniciales
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 200))
    sure_fg = cv2.erode(gray, kernel, iterations=1)
    sure_fg = cv2.dilate(sure_fg, kernel, iterations=1)  # Fondo seguro
    _, sure_fg = cv2.threshold(sure_fg, 30, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibujar los contornos en la imagen original
    img_with_contours = img.copy()
    cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)

    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    # Extraer la región dentro de los contornos
    region_inside = cv2.bitwise_and(gray, gray, mask=mask)

    grad_mag = cv2.Sobel(region_inside, cv2.CV_64F, 1, 0, ksize=5)
    grad_mag = np.uint8(np.clip(grad_mag, 0, 255))  # Magnitud del gradiente
    grad_mag = cv2.convertScaleAbs(grad_mag, alpha=2.0)

    s = np.linspace(0, 2 * np.pi, 400)
    r = grad_mag.shape[0]//2 + grad_mag.shape[0]//1.6 * np.sin(s)
    c = grad_mag.shape[1]//2 + grad_mag.shape[0]//1.8 * np.cos(s)
    init = np.array([r, c]).T
    
    # Aplicar el algoritmo de Snakes (Active Contour)
    image_float = img_as_float(grad_mag)
    snake = active_contour(image_float, init, alpha=0.06, beta=20, gamma=0.01)

    # Visualización de resultados
    plt.figure(figsize=(16, 12))

    plt.subplot(2, 3, 1)
    plt.title("Imagen Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Eq")
    plt.imshow(gray, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Primer Plano Seguro")
    plt.imshow(sure_fg, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title("Contornos Finales en Imagen")
    plt.imshow(cv2.cvtColor(img_with_contours, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title("Gradiente de Intensidad")
    plt.imshow(grad_mag, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("Líneas Detectadas")
    plt.imshow(img, cmap="gray")
    plt.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    plt.axis('off')

    plt.show()

    return img_with_contours


# Ruta de la imagen
image_path = "Material Mama/Glandular-graso/mdb021.jpg"
img_with_contours = process_with_watershed(image_path)
