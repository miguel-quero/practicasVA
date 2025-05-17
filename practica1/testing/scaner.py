import numpy as np
import cv2
import sys
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

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
            idx = 0  # Cuadrante superior izquierdo. Primera coordenada
        elif x >= ancho / 2 and y < alto / 2:
            idx = 1  # Cuadrante superior derecho. Segunda coordenada
        elif x < ancho / 2 and y >= alto / 2:
            idx = 3  # Cuadrante inferior izquierdo. Ultima coordenada
        else:
            idx = 2  # Cuadrante inferior derecho. Tercera coordenada

        if match.distance < distancias_min[idx]:
            distancias_min[idx] = match.distance
            mejores_por_cuadrante[idx] = punto

    # Si alguno de los cuadrantes no tiene punto asignado, significa que no se detectaron las 4 esquinas.
    if None in mejores_por_cuadrante:
        print("No se pudieron detectar las 4 esquinas.")
        return None

    # Devolver los 4 puntos seleccionados, que deberían corresponder a las esquinas de la hoja.
    return mejores_por_cuadrante

# Esta función carga los descriptores BRIEF de referencia a partir de las coordenadas obtenidas con https://www.robots.ox.ac.uk/~vgg/software/via/
#
# Coordenadas guardadas en coordenadas.txt
#
# La idea es usar estos puntos clave para extraer descriptores BRIEF, que luego servirán como referencia
# para el emparejamiento

def cargar_descriptores_referencia(archivo_coordenadas='coordenadas.txt'):
    # Inicializamos el descriptor BRIEF
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    # Lista donde se irán acumulando todos los descriptores extraídos de cada imagen de referencia.
    descriptores_totales = []

    # Abrimos el archivo de coordenadas en modo lectura.
    with open(archivo_coordenadas, 'r') as file:
        for linea in file:

            # Separamos el nombre del archivo de imagen de la lista de coordenadas.
            nombre_imagen, lista_coordenadas = linea.strip().split(':', 1)
            nombre_imagen = nombre_imagen.strip()
            lista_coordenadas = lista_coordenadas.strip()

            # Eliminamos corchetes y paréntesis, luego separamos por coma y agrupamos pares
            lista_coordenadas = lista_coordenadas.replace('[', '').replace(']', '')
            pares = lista_coordenadas.split('),')
            puntos = []
            for par in pares:
                par = par.replace('(', '').replace(')', '').strip()
                if par:
                    x_str, y_str = par.split(',')
                    x = float(x_str.strip())
                    y = float(y_str.strip())
                    puntos.append((x, y))

            #Debug
            print(nombre_imagen, lista_coordenadas)

            # Creamos los keypoints a partir de las coordenadas sacadas de coordenadas.txt
            # https://stackoverflow.com/questions/29415719/how-do-i-create-keypoints-to-compute-sift
            keypoints = [cv2.KeyPoint(float(x), float(y), 0) for (x, y) in puntos]

            # Calculamos los descriptores BRIEF en las posiciones indicadas.
            _, descriptores = brief.compute(imagen, keypoints)

            # Añadimos los descriptores calculados a la lista de todos los descriptores
            descriptores_totales.extend(descriptores)

    # Convertimos la lista final a un array de Numpy y lo devolvemos.
    return np.array(descriptores_totales)


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
    #descriptores_ref = np.load("descriptores_brief.npy")
    descriptores_ref = cargar_descriptores_referencia()

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

