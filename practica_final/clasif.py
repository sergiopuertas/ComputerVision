import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import joblib  # Para guardar el modelo

# Función para calcular características de textura
def calcular_caracteristicas_completas(img):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    proporcion_brillo = hist[200:].sum() / hist.sum()

    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    homogeneidad = graycoprops(glcm, 'homogeneity')[0, 0]
    energia = graycoprops(glcm, 'energy')[0, 0]
    contraste = graycoprops(glcm, 'contrast')[0, 0]
    entropia = -np.sum(hist / hist.sum() * np.log2(hist / hist.sum() + np.finfo(float).eps))

    desviacion = np.std(img)
    skewness = skew(img.flatten())
    kurtosis_val = kurtosis(img.flatten())

    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    lbp_hist = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2))[0].max()

    return [
        proporcion_brillo, homogeneidad, energia, contraste, entropia,
        desviacion, skewness, kurtosis_val, lbp_hist
    ]


# Función para procesar las imágenes y generar un DataFrame
def procesar_imagenes(base_dir, clases):
    datos = []
    for clase in clases:
        ruta_clase = os.path.join(base_dir, clase)
        archivos = os.listdir(ruta_clase)
        print(f"Procesando clase: {clase}, número de archivos: {len(archivos)}")
        for archivo in archivos:
            ruta_archivo = os.path.join(ruta_clase, archivo)
            if os.path.isfile(ruta_archivo) and archivo.lower().endswith('.jpg'):
                img = cv2.imread(ruta_archivo, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    caracteristicas = calcular_caracteristicas_completas(img)
                    datos.append(caracteristicas + [clase])
                else:
                    print(f"Error al leer la imagen: {ruta_archivo}")
            else:
                print(f"Archivo no válido o no es una imagen: {ruta_archivo}")
    columnas = [
        "proporcion_brillo", "homogeneidad", "energia", "contraste", "entropia",
        "desviacion", "skewness", "kurtosis", "lbp", "clase"
    ]
    return pd.DataFrame(datos, columns=columnas)


# Experimentación con Random Forest
def ajustar_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='f1_weighted', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_


# Clasificación y reporte
def clasificar(dataframe, clases):
    y = dataframe["clase"]
    X = dataframe.drop(["clase"], axis=1)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    confusion_total = np.zeros((len(clases), len(clases)))

    for train_index, test_index in skf.split(X, y):
        if isinstance(X, pd.DataFrame):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        else:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        rf, best_params = ajustar_random_forest(X_train, y_train)
        print(f"Mejores parámetros del Random Forest: {best_params}")

        y_pred = rf.predict(X_test)

        # Verificar que las clases en y_test estén en las etiquetas esperadas
        clases_validas = [clase for clase in clases if clase in y_test.values]
        if len(clases_validas) < len(clases):
            print(f"Advertencia: No todas las clases esperadas están en y_test. Clases presentes: {set(y_test)}")

        # Crear la matriz de confusión solo con las clases válidas
        conf_matrix = confusion_matrix(y_test, y_pred, labels=clases_validas)
        confusion_total[:len(clases_validas), :len(clases_validas)] += conf_matrix

        print("=== Reporte de Clasificación ===")
        print(classification_report(y_test, y_pred, target_names=clases_validas))

    return confusion_total, rf.feature_importances_, rf, best_params


# Visualizar Matriz de Confusión
def visualizar_matriz(conf_matrix, clases, titulo):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="f", cmap="Blues", cbar=False, xticklabels=clases, yticklabels=clases)
    plt.title(titulo)
    plt.ylabel("Clase Real")
    plt.xlabel("Clase Predicha")
    plt.show()


# Mostrar las importancias de las características
def visualizar_importancia_caracteristicas(importancias, features):
    df_importancia = pd.DataFrame({
        'Característica': features,
        'Importancia': importancias
    })
    df_importancia = df_importancia.sort_values(by='Importancia', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importancia', y='Característica', data=df_importancia)
    plt.title("Importancia de las Características")
    plt.show()


# Guardar el modelo con mejor performance
def guardar_modelo(modelo, filename):
    joblib.dump(modelo, filename)





# Main
base_dir = "Proc/"
clases = ["Graso", "Glandular-denso", "Glandular-graso"]
df = procesar_imagenes(base_dir, clases)
df_binaria = df.copy()
df_multinomial = df.copy()

# Clasificación Binaria
df_binaria["clase"] = df_binaria["clase"].apply(lambda x: "Graso" if x == "Graso" else "Glandular")
conf_matrix_binaria, importancias_binaria, rf_binaria, best_params_binaria = clasificar(df_binaria.drop(columns=["contraste", "entropia", "kurtosis", "energia", "homogeneidad"]), ["Graso", "Glandular"])
visualizar_matriz(conf_matrix_binaria, ["Graso", "Glandular"], "Matriz de Confusión: Clasificación Binaria")

# Guardar el modelo binario
guardar_modelo(rf_binaria, "modelo_binario.joblib")

# Clasificación Multinomial
df_multinomial = df_multinomial[df_multinomial["clase"] != "Graso"]
conf_matrix_multinomial, importancias_multinomial, rf_multinomial, best_params_multinomial = clasificar(df_multinomial.drop(columns=["entropia", "kurtosis", "skewness", "energia"]), ["Glandular-denso", "Glandular-graso"])
visualizar_matriz(conf_matrix_multinomial, ["Glandular-denso", "Glandular-graso"], "Matriz de Confusión: Clasificación Multinomial")

# Guardar el modelo multinomial
guardar_modelo(rf_multinomial, "modelo_multinomial.joblib")
