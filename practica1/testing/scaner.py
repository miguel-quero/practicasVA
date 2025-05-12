import numpy as np
import cv2
import sys
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

def cargar_coordenadas_txt(archivo_txt):
    """
    Esta función lee el archivo de texto con las coordenadas de las esquinas de las imágenes
    y devuelve un diccionario con el nombre de la imagen como clave y las coordenadas de las esquinas como valor.
    """
    coordenadas = {}
    with open(archivo_txt, 'r') as f:
        for line in f.readlines():
            partes = line.strip().split(": ")
            imagen = partes[0]
            puntos = eval(partes[1])  # Convierte la cadena a una lista de tuplas
            coordenadas[imagen] = puntos
    return coordenadas

# Preproceso de la imagen, eliminar ruido con morfología y umbralización.
def PreprocesarImagen(imagen):
    """
    Esta función convierte la imagen a escala de grises, suaviza el histograma para
    encontrar picos y realiza una umbralización para convertir la imagen en una imagen
    binaria. Luego, aplica un proceso morfológico de erosión y dilatación para limpiar
    el ruido.
    """
    # Convertir imagen a escala de grises
    imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # Calcular histograma con Numpy en lugar de cv2
    histograma = np.histogram(imagenGris, bins=256, range=(0, 256))[0]
    
    # Suavizar el histograma
    histogramaSuavizado = gaussian_filter1d(histograma, sigma=2)

    # Buscar picos en el histograma suavizado
    picos, _ = find_peaks(histogramaSuavizado, distance=20, prominence=100)

    # En el histograma suavizado si hay un pico alto, se utiliza como umbral, si hay dos picos altos, se utiliza el umbral medio
    umbral = 127
    if len(picos) >= 2:
        picos = sorted(picos, key=lambda x: histogramaSuavizado[x], reverse=True)
        umbral = (picos[0] + picos[1]) / 2
    elif len(picos) == 1:
        umbral = picos[0]

    # Umbralizar imagen -> Imagen bitonal
    _, imagenUmbralizada = cv2.threshold(imagenGris, int(umbral), 255, cv2.THRESH_BINARY)

    # Aplicar morfología (erosión y dilatación) para limpiar ruido
    kernel = np.ones((3, 3), np.uint8)
    imagenPreprocesada = cv2.erode(imagenUmbralizada, kernel)
    imagenPreprocesada = cv2.dilate(imagenPreprocesada, kernel)

    return imagenPreprocesada


# Usando Harris o FAST, detectar puntos de interés y calcular descriptores
def DetectarEsquinas(imagen):
    """
    Esta función detecta las esquinas en la imagen utilizando el algoritmo Harris.
    Devuelve la imagen con las esquinas detectadas dibujadas sobre ella.
    """
    # Ya la recibimos en gris
    threshold = 0.05
    blockSize = 7  # Tamaño de la ventana
    ksize = 7  # Tamaño del kernel de Sobel
    k = 0.05  # Factor de Harris

    # Detectar esquinas con Harris
    esquinas = cv2.cornerHarris(imagen, blockSize, ksize, k)

    # Umbral para detectar las esquinas más destacadas
    indices = esquinas > threshold * esquinas.max()  # Filtrar las esquinas
    imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    print(f"Esquinas detectadas: {len(indices)}")

    # Coordenadas de las esquinas
    coords = [(j, i) for i in range(0, indices.shape[0]) for j in range(0, indices.shape[1]) if indices[i, j]]

    # Dibujar las esquinas detectadas en la imagen
    for coord in coords:
        cv2.circle(imagen, coord, 3, (0, 255, 0), -1)  # Dibuja los puntos de esquina

    return imagen  # Devuelve la imagen con las esquinas dibujadas


# Usando BFMatcher, elegir el punto singular de cada cuadrante de la imagen de test que minimice la distancia a algún descriptor de referencia
def EmparejarEsquinas(img1, img2):
    """
    Esta función usa ORB para detectar descriptores de puntos de interés en las dos imágenes.
    Luego, empareja estos descriptores utilizando BFMatcher y la prueba de relación para encontrar
    los emparejamientos más cercanos entre las dos imágenes.
    """
    # Detector ORB
    sift = cv2.ORB_create()

    # Detectar descriptores con ORB
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # BFMatcher
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio
    good = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good.append([m])

    # cv2.drawMatchesKnn espera una lista de listas como coincidencias
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    return img3


# Rectificación de la imagen usando una transformación de perspectiva
# Orden de las esquinas: [superior izquierda, superior derecha, inferior derecha, inferior izquierda]
def RectificarImagen(imagen, esquinas):
    """
    Esta función rectifica la imagen usando una transformación de perspectiva basada en las esquinas
    detectadas. Utiliza las coordenadas de las esquinas para aplicar una transformación de perspectiva
    y obtener una imagen con un tamaño proporcional a un A4.
    """
    # Transformar las esquinas a un array numpy tipo float32 para la funcion getPerspectiveTransform
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
    """
    Esta función muestra una imagen redimensionada en una ventana de OpenCV.
    Recibe el título para la ventana y la imagen que se va a mostrar, así como un factor de escala.
    """
    alto, ancho = imagen.shape[:2]
    redimensionada = cv2.resize(imagen, (int(ancho * escala), int(alto * escala)))
    cv2.imshow(titulo, redimensionada)


# Main
if __name__ == "__main__":
    """
    Esta es la función principal que se ejecuta cuando se corre el script. Aquí se lee la imagen de entrada,
    se cargan las coordenadas de las esquinas desde un archivo de texto, se preprocesa la imagen, se rectifica
    y finalmente se muestra y guarda la imagen resultante.
    """
    # Comprobar que se pasa el nombre de la imagen como argumento
    if len(sys.argv) != 2:
        print("Error. Ejemplo: python scaner.py imagenEntrada.jpg")
        exit()

    imagenEntrada = sys.argv[1]
    imagenSalida = "salida.jpg"

    # Leer las coordenadas desde el archivo de texto
    coordenadas = cargar_coordenadas_txt("coordenadas.txt")

    # Comprobar que la imagen tiene coordenadas disponibles
    if imagenEntrada not in coordenadas:
        print(f"No se encuentran coordenadas para {imagenEntrada}")
        exit()

    # Leer la imagen
    imagen = cv2.imread(imagenEntrada)

    if imagen is None:
        print("Imagen no válida o no cargada.")
        exit()

    # Preprocesar la imagen
    imagenPreprocesada = PreprocesarImagen(imagen)

    # Obtener las coordenadas de la imagen actual
    coordenadas_imagen = coordenadas[imagenEntrada]

    # Rectificar la imagen utilizando las coordenadas
    imgRectificada = RectificarImagen(imagen, coordenadas_imagen)

    # Mostrar las imágenes
    mostrar_redimensionada('imagen', imagen)
    mostrar_redimensionada('imagenPreprocesada', imagenPreprocesada)
    mostrar_redimensionada('imgRectificada', imgRectificada)

    # Guardar la imagen rectificada
    cv2.imwrite(imagenSalida, imgRectificada)
    print("Imagen guardada:", imagenSalida)

    cv2.waitKey(0)  # Espera infinita hasta que pulses una tecla
    cv2.destroyAllWindows()  # Cerrar todas las ventanas de imágenes




