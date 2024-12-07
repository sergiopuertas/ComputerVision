import cv2
import numpy as np
import joblib
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew, kurtosis


# Cargar el modelo guardado
def cargar_modelo(filename):
    return joblib.load(filename)


def calcular_caracteristicas_completas(img, binario=True):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    proporcion_brillo = hist[200:].sum() / hist.sum()

    glcm = graycomatrix(img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    homogeneidad = graycoprops(glcm, 'homogeneity')[0, 0]
    contraste = graycoprops(glcm, 'contrast')[0, 0]

    desviacion = np.std(img)
    skewness = skew(img.flatten())
    kurtosis_val = kurtosis(img.flatten())

    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(img, n_points, radius, method='uniform')
    lbp_hist = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2))[0].max()

    # Características para clasificación binaria (con menos parámetros)
    if binario:
        return [
            proporcion_brillo, skewness,
            desviacion, lbp_hist
        ]
    # Características para clasificación multinomial (con más parámetros)
    else:
        return [
            proporcion_brillo, homogeneidad, contraste,
            desviacion, lbp_hist
        ]


# Función para procesar y predecir una imagen
def predecir_imagen(imagen_path, modelo, binario=True):
    img = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {imagen_path}")

    # Calcular características
    caracteristicas = calcular_caracteristicas_completas(img, binario)
    caracteristicas = np.array(caracteristicas).reshape(1, -1)

    # Realizar la predicción
    prediccion = modelo.predict(caracteristicas)
    return prediccion[0]


if __name__ == "__main__":
    # Cargar el modelo para clasificación binaria
    modelo_binario = cargar_modelo("modelo_binario.joblib")

    # Probar con una imagen
    imagen_path = "Proc/Graso/mdb005.jpg"

    # Realizar la predicción binaria
    prediccion_binaria = predecir_imagen(imagen_path, modelo_binario)

    if prediccion_binaria == "Glandular":
        # Si es "Glandular", cargar el modelo multinomial para las subclases
        modelo_multinomial = cargar_modelo("modelo_multinomial.joblib")
        prediccion_multinomial = predecir_imagen(imagen_path, modelo_multinomial, binario=False)
        print(f"Predicción multinomial: {prediccion_multinomial}")
        prediccion = prediccion_multinomial
    else:
        prediccion = prediccion_binaria

    print(f"La clase predicha para la imagen es: {prediccion}")
