import cv2
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from skimage.feature import graycomatrix, graycoprops

def process(img):
    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    edge_percentage = np.sum(cv2.Canny(img, 100, 200)) / (img.shape[0] * img.shape[1])

    data = {
        'edge_percentage': edge_percentage,
        'glcm_dissimilarity': dissimilarity,
        'glcm_contrast': contrast,
    }

    return img, data

def extract_features(image_dir, base_dir, classes):
    features = []
    labels = []

    for label, class_name in enumerate(classes):
        image_path = os.path.join(image_dir, class_name)
        mask_dir = os.path.join(base_dir, class_name)

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
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                img = cv2.bitwise_and(img, img, mask=mask)
                img = cv2.equalizeHist(img)
                if img is not None:
                    _, data = process(img)
                    features.append(data)
                    labels.append(label)
                else:
                    print(f"No se pudo cargar la imagen: {image_path}")

    return pd.DataFrame(features), labels

def simple_classification(data):
    if data['glcm_contrast'] <= 42:
        if data['glcm_dissimilarity'] <= 1:
            if data['glcm_contrast'] <= 30:
                return "Glandular-denso"
            else:
                return "Glandular-graso"
        else:
            return "Glandular-denso"
    else:
        if data['glcm_contrast'] <= 49:
            return "Glandular-graso"
        else:
            if data['edge_percentage'] <= 8:
                return "Graso"
            else:
                return "Glandular-graso"

def classify_images(features):
    predictions = []

    for _, feature in features.iterrows():
        pred = simple_classification(feature)
        predictions.append(pred)

    return predictions

def classify_single_image(image_path):
    mask_dir = "ProcMM"
    image_dir = "Material Mama"
    mask = os.path.join(mask_dir, image_path)
    image = os.path.join(image_dir, image_path)
    img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    img = cv2.bitwise_and(img, img, mask=mask)
    img = cv2.equalizeHist(img)
    _, data = process(img)

    prediction = simple_classification(data)
    return prediction

def calculate_metrics(labels, predictions, classes):
    labels_names = [classes[label] for label in labels]

    # Reporte de clasificación
    report = classification_report(labels_names, predictions, target_names=classes)
    print("\nReport de Clasificación:")
    print(report)

if __name__ == "__main__":

    image_dir = "Material Mama/"
    mask_dir = "ProcMM"
    classes = ["Glandular-denso", "Glandular-graso", "Graso"]

    features, labels = extract_features(image_dir, mask_dir, classes)

    if not features.empty:
        predictions = classify_images(features)
        calculate_metrics(labels, predictions, classes)
    else:
        print("No se encontraron características para clasificar.")

    try:
        image_path = "Glandular-denso/mdb107.jpg"
        prediction = classify_single_image(image_path)
        print(f"La predicción para la imagen es: {prediction}")
    except ValueError as e:
        print(e)
