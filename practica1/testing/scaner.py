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
def DetectarEsquinas(image):
    return keypoints, descriptors

# Usando BFMatcher, elegir el punto singular de cada cuadrante de la imagen de test que minimice la distancia a algún descriptor de referencia
def EmparejarEsquinas(test_descriptors, reference_descriptors):
    return best_points


# Recorte y rectificación del papel
def RectificarImagen(imagen, esquinas):
    return imagen

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
    imagenFinal = PreprocesarImagen(imagen)
    mostrar_redimensionada('Resultado', imagenFinal)
    cv2.waitKey(0)

    # Guardar la imagen final
    #cv2.imwrite(imagenSalida, imagenFinal)
    #print("Imagen guardada: ", imagenSalida)
