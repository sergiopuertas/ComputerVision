import cv2
import os
import pandas as pd

from sklearn.metrics import confusion_matrix, classification_report
from skimage.feature import graycomatrix, graycoprops
# Procesar la imagen para extraer características
def process(image):
    img = image
    # Calcular estadísticas de intensidad
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    pixel_values = img.ravel()
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    kurtosis = pd.Series(pixel_values).kurtosis()

    # Crear un diccionario con las características calculadas
    data = {
        'glcm_dissimilarity': dissimilarity,
        'glcm_homogeneity': homogeneity,
        'kurtosis': kurtosis,
        'glcm_contrast': contrast,
    }

    return img, data

# Extraer características y etiquetas
def extract_features(image_dir,base_dir, classes):
    features = []
    labels = []

    for label, class_name in enumerate(classes):
        image_path = os.path.join(image_dir, class_name)
        mask_dir = os.path.join(base_dir, class_name)
        mask_dir = os.path.join(mask_dir, "mask2")

        if not os.path.exists(mask_dir):
            print(f"Directorio no encontrado: {mask_dir}")
            continue

        for file_name in os.listdir(mask_dir):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                mask = os.path.join(mask_dir, file_name)
                image = os.path.join(image_path, file_name)
                img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))  # Ensure mask is the same size as img
                img = cv2.bitwise_and(img, img, mask=mask)
                img = cv2.equalizeHist(img)
                if img is not None:
                    _, data = process(img)
                    features.append(data)
                    labels.append(label)
                else:
                    print(f"No se pudo cargar la imagen: {image_path}")

    return pd.DataFrame(features), labels

# Función para la clasificación binaria utilizando ifs
def simple_classification(data):
    if data['glcm_dissimilarity'] > 1.4 and data['glcm_homogeneity'] > 0.78:
       return 'Graso'
    else:
        if data['glcm_contrast'] <= 24:
            return "Glandular-denso"
        else:
            if data['kurtosis'] <= 0.2:
                return "Glandular-denso"
            else:
                return "Glandular-graso"


# Clasificar imágenes
def classify_images(features):
    predictions = []

    for _, feature in features.iterrows():
        pred = simple_classification(feature)
        predictions.append(pred)

    return predictions

# Calcular métricas
def calculate_metrics(labels, predictions, classes):
    # Convertir etiquetas a nombres de clases
    labels_names = [classes[label] for label in labels]

    # Matriz de confusión
    cm = confusion_matrix(labels_names, predictions, labels=classes)
    print("\nMatriz de Confusión:")
    print(cm)

    # Reporte de clasificación
    report = classification_report(labels_names, predictions, target_names=classes)
    print("\nReporte de Clasificación:")
    print(report)

# Ejemplo de uso
if __name__ == "__main__":

    image_dir = "Material Mama/"
    mask_dir = "ProcMM"
    classes = ["Glandular-denso", "Glandular-graso", "Graso"]

    features, labels = extract_features(image_dir,mask_dir, classes)

    # Clasificar las imágenes
    if not features.empty:
        predictions = classify_images(features)

        # Mostrar resultados
        for label, pred in zip(labels, predictions):
            print(f"Clase real: {classes[label]} - Predicción: {pred}")

        # Calcular y mostrar métricas
        calculate_metrics(labels, predictions, classes)
    else:
        print("No se encontraron características para clasificar.")
