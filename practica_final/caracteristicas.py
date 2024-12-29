from skimage.feature import graycomatrix, graycoprops
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_selection import SelectKBest, f_classif
import pandas as pd
import numpy as np
import cv2
import os
from scipy.stats import skew
import matplotlib.pyplot as plt

# Función para extraer características GLCM
def extract_glcm_features(image):
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    return {
        'glcm_contrast': contrast,
        'glcm_dissimilarity': dissimilarity,
        'glcm_homogeneity': homogeneity,
        'glcm_energy': energy,
    }

# Procesar la imagen para extraer características
def process(image):
    # Asegurar que la imagen esté en escala de grises
    img = image

    # Calcular estadísticas de intensidad
    pixel_values = img.ravel()
    mean_intensity = np.mean(pixel_values)
    std_intensity = np.std(pixel_values)
    skewness = skew(pixel_values)
    dynamic_range = np.max(pixel_values) - np.min(pixel_values)
    kurtosis = pd.Series(pixel_values).kurtosis()

    edges = cv2.Canny(img, 100, 200)
    edge_percentage = np.sum(edges) / (edges.shape[0] * edges.shape[1])

    # Extraer características adicionales
    glcm_features = extract_glcm_features(img)

    # Agregar las propiedades calculadas a los datos de salida
    data = {
        'mean_intensity': mean_intensity,
        'std_intensity': std_intensity,
        'skewness': skewness,
        'dynamic_range': dynamic_range,
        'edge_percentage': edge_percentage,
        'kurtosis': kurtosis,
        **glcm_features,
    }

    return img, data

# Función para extraer características de las imágenes y las etiquetas
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
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))  # Ensure mask is the same size as img
                img = cv2.bitwise_and(img, img, mask=mask)
                img = cv2.equalizeHist(img)
                if img is not None:
                    _, data = process(img)
                    features.append(data)
                    labels.append(label)  # Multiclase: 0, 1, 2
                else:
                    print(f"No se pudo cargar la imagen: {image_path}")

    return pd.DataFrame(features), labels

# Buscar la mejor combinación de características
def optimize_features(features, labels, max_depth=3):
    best_score = 0
    best_features = None
    best_model = None

    # Selección de características
    for k in range(1, features.shape[1] + 1):
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(features, labels)
        selected_features = selector.get_support(indices=True)

        # Entrenar el modelo
        clf = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
        clf.fit(X_selected, labels)

        # Evaluar el modelo
        y_pred = clf.predict(X_selected)
        score = f1_score(labels, y_pred, average='weighted')

        if score > best_score:
            best_score = score
            best_features = selected_features
            best_model = clf

    # Mostrar resultados
    print(f"Mejor puntaje F1: {best_score}")
    final_feature_names = features.columns[best_features]
    print(f"Mejores características: {list(final_feature_names)}")

    # Mostrar las reglas del mejor árbol
    tree_rules = export_text(best_model, feature_names=list(final_feature_names))
    print("\nReglas del Árbol de Decisión:\n")
    print(tree_rules)

    # Visualizar el mejor árbol
    plt.figure(figsize=(12, 8))
    plot_tree(best_model, feature_names=list(final_feature_names), class_names=[str(c) for c in set(labels)], filled=True)
    plt.show()

    return best_model, final_feature_names

if __name__ == "__main__":
    image_dir = "Material Mama/"
    mask_dir = "ProcMM"
    classes = ["Glandular-denso", "Glandular-graso", "Graso"]

    features, labels = extract_features(image_dir, mask_dir, classes)

    if not features.empty:
        optimize_features(features, labels)
    else:
        print("No se encontraron características para clasificar.")
