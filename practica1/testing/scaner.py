import numpy as np
import cv2
import sys
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

# Preproceso de la imagen, eliminar ruido con morfologia y umbralización.
def PreprocesarImagen(imagen):
    
    # Convertir imagen a escala de grises
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Calcular histograma con Numpy en lugar de cv2
    # Incluye del 0 al 255, el 256 no.
    histograma = np.histogram(imagenGris, bins = 256, range = (0, 256))[0]
    
    # Suavizar el histograma
    # Sigma es 2 para un suavizado medio mas o menos
    histogramaSuavizado = gaussian_filter1d(histograma, sigma=2)

    # Buscar picos en el histograma suavizado
    # Distancia minima de 20 niveles de gris entre picos separados
    # Altura minima de 100 para detectar picos.
    picos, _ = find_peaks(histogramaSuavizado, distance = 20, prominence = 100)

    # En el histograma suavizado si hay un pico alto, se utiliza como umbral, si hay dos picos altos, se utiliza el umbral medio
    umbral = 127
    if len(picos) >= 2:
        picos = sorted(picos, key = lambda x: histogramaSuavizado[x], reverse = True)
        umbral = (picos[0] + picos[1]) / 2

    elif len(picos) == 1:
        umbral = picos[0]

    # Umbralizar imagen -> Imagen bitonal
    _, imagenUmbralizada = cv2.threshold(imagenGris, umbral, 255, cv2.THRESH_BINARY)
    
    # Morfología para limpiar ruido, erosión y dilatación con kernel 3x3
    kernel = np.ones((3,3), np.uint8)
    imagenPreprocesada = cv2.erode(imagenUmbralizada, kernel)
    imagenPreprocesada = cv2.dilate(imagenPreprocesada, kernel)

    return imagenPreprocesada


# Usando Harris o FAST, detectar puntos de interés y calcular descriptores
def DetectarEsquinas(imagen):
    # Ya la recibimos en gris
    # Parametros de Harris
    threshold = 0.05
    blockSize = 7  # Tamaño de la ventana
    ksize = 7  # Tamaño del kernel de Sobel
    k = 0.05  # Factor de Harris

    # Detectar esquinas con Harris
    esquinas = cv2.cornerHarris(imagen, blockSize, ksize, k)

    # Umbral para detectar las esquinas más destacadas
    indices = esquinas > threshold * esquinas.max()  # Filtrar las esquinas
    imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
    print(f"Esquinas detectadas: {len(indices)}")

    # Coordenadas de las esquinas
    coords = [(j, i) for i in range(0, indices.shape[0]) for j in range(0, indices.shape[1]) if indices[i, j]]

    # Dibujar las esquinas detectadas en la imagen
    for coord in coords:
        cv2.circle(imagen, coord, 3, (0, 255, 0), -1)  # Dibuja los puntos de esquina

    return imagen  # Devuelve la imagen con las esquinas dibujadas

# Usando BFMatcher, elegir el punto singular de cada cuadrante de la imagen de test que minimice la distancia a algún descriptor de referencia
def EmparejarEsquinas(test_descriptors, reference_descriptors):
    return best_points


# Rectificación de la imagen usando una transformación de perspectiva
# Orden de las esquinas: [superior izquierda, superior derecha, inferior derecha, inferior izquierda]
def RectificarImagen(imagen, esquinas):

    # Transformar las esquinas a un array numpy tipo float32 para la funcion getPerspectiveTransform
    srcImagen = np.array(esquinas,np.float32)

    # Calcular el ancho similar a la distancia entre las dos esquinas superiores detectadas.
    # Funcion de numpy para calcular la distancia entre dos puntos, para evitar utilizar la fórmula de distancia euclídea
    # Fuente: https://numpy.org/doc/2.2/reference/generated/numpy.linalg.norm.html
    ancho = int(np.linalg.norm(srcImagen[0] - srcImagen[1]))

    # Calcular alto de tamaño proporcional a un A4 -> proporción => 1:√2 -> Alto / Ancho = √2 -> Alto = Ancho * √2
    # Fuente: https://estudiesteve.es/blog/29-din-a4-medidas-ventajas-e-historia-del-formato
    alto = int(ancho * np.sqrt(2))

    # Esquinas de destino de la hoja de la imagen
    dstImagen = np.array([
        [0, 0],
        [ancho, 0],
        [ancho, alto],
        [0, alto]
    ],np.float32)

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
    imagenHarris = DetectarEsquinas(imagenPreprocesada)

    mostrar_redimensionada('Resultado', imagenHarris)
    cv2.waitKey(0)

    # Guardar la imagen final
    #cv2.imwrite(imagenSalida, imagenFinal)
    #print("Imagen guardada: ", imagenSalida)
