import cv2
import os
import pandas as pd
import numpy as np


# Procesar la imagen para extraer características
def process(image):
    img = image
    # Calcular estadísticas de intensidad
    pixel_values = img.ravel()
    std_intensity = np.std(pixel_values)
    dynamic_range = np.max(pixel_values) - np.min(pixel_values)
    edges = cv2.Canny(img, 100, 200)
    edge_percentage = np.sum(edges) / (edges.shape[0] * edges.shape[1])

    # Crear un diccionario con las características calculadas
    data = {
        'std_intensity': std_intensity,
        'dynamic_range': dynamic_range,
        'edge_percentage': edge_percentage,
    }

    return img, data


# Extraer características y etiquetas
def extract_features(base_dir, classes):
    features = []
    labels = []

    for label, class_name in enumerate(classes):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Directorio no encontrado: {class_dir}")
            continue

        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(class_dir, file_name)
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    _, data = process(img)
                    features.append(data)
                    labels.append(label)
                else:
                    print(f"No se pudo cargar la imagen: {file_path}")

    return pd.DataFrame(features), labels


# Función para la clasificación binaria utilizando ifs
def simple_classification(data):
    std_intensity = data['std_intensity']
    dynamic_range = data['dynamic_range']
    edge_percentage = data['edge_percentage']

    # Árbol binario con condiciones if
    if std_intensity <= 74.70:
        if edge_percentage <= 0.44:
            if dynamic_range <= 183.50:
                return 'Graso'
            else:
                return 'Glandular-denso'
        else:
            if edge_percentage <= 0.53:
                return 'Glandular-graso'
            else :
                return 'Graso'
    else:
        if edge_percentage <= 0.68:
            return 'Glandular-denso'
        else:
            return 'Glandular-graso'


# Clasificar imágenes
def classify_images(features):
    predictions = []

    for _, feature in features.iterrows():
        pred = simple_classification(feature)
        predictions.append(pred)

    return predictions


# Ejemplo de uso
if __name__ == "__main__":

    base_dir = "ProcMM"
    classes = ["Glandular-denso", "Glandular-graso", "Graso"]

    features, labels = extract_features(base_dir, classes)

    # Clasificar las imágenes
    if not features.empty:
        predictions = classify_images(features)

        # Mostrar resultados
        for label, pred in zip(labels, predictions):
            print(f"Clase real: {classes[label]} - Predicción: {pred}")
    else:
        print("No se encontraron características para clasificar.")
