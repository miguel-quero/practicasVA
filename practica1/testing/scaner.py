import numpy as np
import cv2
import sys
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# Preproceso de la imagen, eliminar ruido con morfología y umbralización.
def PreprocesarImagen(imagen):
    # Convertir imagen a escala de grises
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Calcular histograma con Numpy en lugar de cv2
    # Incluye del 0 al 255, el 256 no.
    histograma = np.histogram(imagenGris, bins=256, range=(0, 256))[0]

    # Suavizar el histograma
    # Sigma es 2 para un suavizado medio más o menos
    histogramaSuavizado = gaussian_filter1d(histograma, sigma=2)

    # Buscar picos en el histograma suavizado
    # Distancia mínima de 20 niveles de gris entre picos separados
    # Altura mínima de 100 para detectar picos.
    picos, _ = find_peaks(histogramaSuavizado, distance=20, prominence=100)

    # En el histograma suavizado si hay un pico alto, se utiliza como umbral,
    # si hay dos picos altos, se utiliza el umbral medio
    umbral = 127
    if len(picos) >= 2:
        picos = sorted(picos, key=lambda x: histogramaSuavizado[x], reverse=True)
        umbral = (picos[0] + picos[1]) / 2
    elif len(picos) == 1:
        umbral = picos[0]

    # Umbralizar imagen -> Imagen bitonal
    _, imagenUmbralizada = cv2.threshold(imagenGris, int(umbral), 255, cv2.THRESH_BINARY)

    # Morfología para limpiar ruido, erosión y dilatación con kernel 3x3
    kernel = np.ones((3, 3), np.uint8)
    imagenPreprocesada = cv2.erode(imagenUmbralizada, kernel)
    imagenPreprocesada = cv2.dilate(imagenPreprocesada, kernel)

    return imagenPreprocesada

def DetectarEsquinasAutomaticas(imagen_preprocesada, descriptores_referencia):
    # Detectar puntos clave usando FAST, adecuado para combinar con BRIEF (descriptor binario)
    fast = cv2.FastFeatureDetector_create()
    keypoints = fast.detect(imagen_preprocesada, None)

    # Calcular descriptores BRIEF en los puntos detectados
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    keypoints, descriptores = brief.compute(imagen_preprocesada, keypoints)

    if descriptores is None or len(keypoints) == 0:
        print("No se encontraron descriptores.")
        return None

    # Según el Tema 3  : Detección y descripción de puntos de interés, en la diapositiva 99
    # para emparejar descriptores binarios (como BRIEF o ORB)
    # se utiliza BFMatcher con distancia de Hamming.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Se aplica knnMatch con k=2 para obtener los dos mejores matches por descriptor,
    # siguiendo el método de emparejamiento mostrado en la diapositiva 99del tema 3,
    # lo que permite aplicar el ratio test de Lowe para mejorar la calidad del emparejamiento.
    matches_knn = bf.knnMatch(descriptores, descriptores_referencia, k=2)

    # Aplicamos el ratio test de Lowe (0.75 es un valor típico) para filtrar matches ambiguos o falsos positivos.
    good_matches = []
    for m, n in matches_knn:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Se calcula el tamaño de la imagen para dividirla en cuadrantes, con el objetivo de seleccionar
    # un punto singular representativo en cada cuadrante, como exige el enunciado.
    alto, ancho = imagen_preprocesada.shape[:2]

    # Inicializamos la selección de los mejores matches por cuadrante
    mejores_por_cuadrante = [None, None, None, None]  # TL, TR, BL, BR
    distancias_min = [float('inf')] * 4

    # Para cada good match, clasificamos su punto en uno de los 4 cuadrantes de la imagen
    # y seleccionamos el match con menor distancia (más fiable) por cuadrante.
    for match in good_matches:
        punto = keypoints[match.queryIdx].pt
        x, y = punto

        if x < ancho / 2 and y < alto / 2:
            idx = 0  # Cuadrante superior izquierdo
        elif x >= ancho / 2 and y < alto / 2:
            idx = 1  # Cuadrante superior derecho
        elif x < ancho / 2 and y >= alto / 2:
            idx = 2  # Cuadrante inferior izquierdo
        else:
            idx = 3  # Cuadrante inferior derecho

        if match.distance < distancias_min[idx]:
            distancias_min[idx] = match.distance
            mejores_por_cuadrante[idx] = punto

    # Si alguno de los cuadrantes no tiene punto asignado, significa que no se detectaron las 4 esquinas.
    if None in mejores_por_cuadrante:
        print("No se pudieron detectar las 4 esquinas.")
        return None

    # Devolver los 4 puntos seleccionados, que deberían corresponder a las esquinas de la hoja.
    return mejores_por_cuadrante


# Rectificación de la imagen usando una transformación de perspectiva
# Orden de las esquinas: [superior izquierda, superior derecha, inferior derecha, inferior izquierda]
def RectificarImagen(imagen, esquinas):
    # Transformar las esquinas a un array numpy tipo float32 para la función getPerspectiveTransform
    srcImagen = np.array(esquinas, np.float32)

    # Calcular el ancho similar a la distancia entre las dos esquinas superiores detectadas.
    ancho = int(np.linalg.norm(srcImagen[0] - srcImagen[1]))

    # Calcular alto de tamaño proporcional a un A4 -> proporción => 1:√2 -> Alto / Ancho = √2 -> Alto = Ancho * √2
    alto = int(ancho * np.sqrt(2))

    # Esquinas de destino de la hoja de la imagen
    dstImagen = np.array([
        [0, 0],
        [ancho, 0],
        [ancho, alto],
        [0, alto]
    ], np.float32)

    # Calcular la matriz de transformación
    matrizPerspectiva = cv2.getPerspectiveTransform(srcImagen, dstImagen)

    # Transformación de la imagen
    return cv2.warpPerspective(imagen, matrizPerspectiva, (ancho, alto))

# Prueba mostrar imagen
def mostrar_redimensionada(titulo, imagen, escala=0.2):
    alto, ancho = imagen.shape[:2]
    redimensionada = cv2.resize(imagen, (int(ancho * escala), int(alto * escala)))
    cv2.imshow(titulo, redimensionada)

# Main
if __name__ == "__main__":
    # Comprobar los parámetros
    if len(sys.argv) != 3:
        print("Error. Ejemplo: python scaner.py imagenEntrada.jpg imagenSalida.jpg")
        exit()

    # Nombres de las imágenes entrada y salida
    imagenEntrada = sys.argv[1]
    imagenSalida = sys.argv[2]

    # Lectura Imagen
    imagen = cv2.imread(imagenEntrada)

    # Comprobar imagen cargada
    if imagen is None:
        print("Imagen no válida o no cargada.")
        exit()

    # Preproceso de la imagen
    imagenPreprocesada = PreprocesarImagen(imagen)

    # Cargar descriptores de referencia
    print(" Cargando descriptores de referencia...")
    descriptores_ref = np.load("descriptores_brief.npy")

    # Detectar esquinas automáticamente
    print(" Detectando esquinas automáticamente...")
    esquinas = DetectarEsquinasAutomaticas(imagenPreprocesada, descriptores_ref)

    if esquinas is None:
        print(" No se pudieron detectar las 4 esquinas necesarias.")
        exit()

    print(f"Esquinas detectadas: {esquinas}")

    # Rectificar imagen usando las esquinas detectadas
    print("⏳ Rectificando imagen...")
    imagenRectificada = RectificarImagen(imagen, esquinas)

    # Mostrar imágenes para debug
    mostrar_redimensionada('Imagen original', imagen)
    mostrar_redimensionada('Imagen preprocesada', imagenPreprocesada)
    mostrar_redimensionada('Imagen rectificada', imagenRectificada)

    # Guardar imagen rectificada
    cv2.imwrite(imagenSalida, imagenRectificada)
    print(f" Imagen rectificada guardada como: {imagenSalida}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

