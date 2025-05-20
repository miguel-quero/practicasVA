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

    # Morfología para limpiar ruido, erosión y dilatación con kernel 3x3. Apertura
    kernel = np.ones((3, 3), np.uint8)

    # Hemos utilizado un filtro de suavizado gaussiano para eliminar un poco el ruido porque así nos detectaba mejor las esquinas con Harris y teníamos mejor rectificadas las imagenes
    imagenPreprocesada = cv2.GaussianBlur(imagenUmbralizada, (3, 3), 1)

    imagenPreprocesada = cv2.erode(imagenPreprocesada, kernel)
    imagenPreprocesada = cv2.dilate(imagenPreprocesada, kernel)
    imagenPreprocesada = cv2.dilate(imagenPreprocesada, kernel)
    

    return imagenPreprocesada

# Esta función carga los descriptores BRIEF de referencia a partir de las coordenadas
    # obtenidas con el software  VIA https://www.robots.ox.ac.uk/~vgg/software/via/
    #
    # Coordenadas guardadas en coordenadas.txt
    #
    # La idea es usar estos puntos clave para extraer descriptores BRIEF, que luego servirán
    # como referencia para el emparejamiento
def CargarDescriptoresReferencia(rutaCoordenadas):
    

    # Inicializamos el descriptor BRIEF
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

    # Lista donde se irán acumulando todos los descriptores extraídos de las imagenes de entrenamiento
    listaDescriptores = []

    # Abrimos el archivo de coordenadas
    with open(rutaCoordenadas, 'r') as coordenadasTxt:
        for linea in coordenadasTxt:

            # Separamos el nombre del archivo de imagen de la lista de coordenadas.
            nombreImagen,coordenadas = linea.strip().split(':', 1)
            nombreImagen = nombreImagen.strip()
            coordenadas = coordenadas.strip()

            # Eliminamos corchetes y paréntesis, luego separamos por coma y agrupamos las coordenadas en pares
            coordenadas= coordenadas.replace('[', '').replace(']', '')
            pares = coordenadas.split('),')
            puntos = []
            for par in pares:
                par= par.replace('(', '').replace(')', '').strip()
                if par:
                    coordenadaX, coordenadaY= par.split(',')
                    x = float(coordenadaX.strip())
                    y= float(coordenadaY.strip())
                    puntos.append((x, y))

            # Creamos los keypoints a partir de las coordenadas sacadas de coordenadas.txt
            # https://stackoverflow.com/questions/29415719/how-do-i-create-keypoints-to-compute-sift
            puntosClave = [cv2.KeyPoint(float(x), float(y), 0) for (x, y) in puntos]

            # Calculamos los descriptores BRIEF en las posiciones indicadas de cada imagen de entrenamiento en niveles de gris para que los detecte mejor.
            imagen = cv2.imread("learning/"+ nombreImagen)
            if imagen is None:
                print("No se pudo cargar la imagen de entrenamiento:",nombreImagen)
                exit()

            imagenGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            _, descriptores = brief.compute(imagenGris, puntosClave)

            # Añadimos los descriptores calculados a la lista de todos los descriptores
            listaDescriptores.extend(descriptores)

    # Convertimos la lista final a un array de Numpy para que lo utilice bien el detector de esquinas Harris y lo devolvemos.
    return np.array(listaDescriptores)




# Detectar esquinas de las hojas de papel con los descriptores cargados de las imágenes de entrenamiento
def DetectarEsquinas(imagenProcesada, descriptoresReferencia):
    # Ya la recibimos en gris
    # Parámetros de Harris
    umbral = 0.05           # Umbral para filtrar las esquinas
    blocksize = 7           # Tamaño de la ventana
    ksize = 7               # Tamaño del kernel de Sobel
    k = 0.05                # Parámetro libre de la ecuación de Harris

    # Detectar esquinas con Harris
    esquinas = cv2.cornerHarris(imagenProcesada, blocksize, ksize, k)

    # Umbral para detectar las esquinas más destacadas
    esquinas = esquinas > umbral * esquinas.max()  # Filtrar las esquinas por el umbral

    # Coordenadas de las esquinas
    coordenadas = [(j, i) for i in range(0, esquinas.shape[0]) for j in range(0, esquinas.shape[1]) if esquinas[i, j]]
    puntosClave = [cv2.KeyPoint(float(x), float(y), 0) for (x, y) in coordenadas]

    # Calcular descriptores BRIEF en los puntos detectados con Harris
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    puntosClave, descriptores = brief.compute(imagenProcesada, puntosClave)

    if descriptores is None or len(puntosClave) == 0:
        print("No se han encontrado descriptores o puntos clave.")
        exit()

    # Según el Tema 3: Detección y descripción de puntos de interés, en la diapositiva 99
    # para emparejar descriptores binarios (como BRIEF o ORB)
    # se utiliza BFMatcher con distancia de Hamming.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # Se aplica knnMatch (Algoritmo K Vecinos más cercanos) con k=2 para obtener los dos mejores matches por descriptor,
    # siguiendo el método de emparejamiento mostrado en la diapositiva 99 del tema 3,
    # lo que permite aplicar el ratio test de Lowe para mejorar la calidad del emparejamiento.
    matches = bf.knnMatch(descriptores, descriptoresReferencia, k=2)

    # Aplicamos el ratio test de Lowe (0.95 es un valor adaptado) para filtrar matches ambiguos o falsos positivos.
    good = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good.append(m)

    # Se calcula el tamaño de la imagen para dividirla en cuadrantes o esquinas, con el objetivo de seleccionar
    # un punto singular representativo en cada esquina, como exige el enunciado.
    alto, ancho = imagenProcesada.shape[:2]

    # Inicializamos la selección de los mejores emparejamientos por esquina
    esquinasFinales = [None]*4 
    distanciasMinimas = [float('inf')] * 4

    # Este algoritmo no es muy complejo, basicamente filtramos los mejores emparejamientos por distancia y diferenciandolos dependiendo de a que esquina pertenecen porque sino es un caos
    # Para cada good match, clasificamos su punto en uno de los 4 cuadrantes de la imagen
    # y seleccionamos el match con menor distancia (más fiable) por cuadrante o esquina.
    for match in good:
        punto = puntosClave[match.queryIdx].pt
        x, y = punto

        if x < ancho / 2 and y < alto / 2:
            esquina = 0  # Cuadrante/ esquina superior izquierda
        elif x >= ancho / 2 and y < alto / 2:
            esquina = 1  # Cuadrante/ esquina superior derecha
        elif x >= ancho / 2 and y >= alto / 2:
            esquina = 2  # Cuadrante/ esquina inferior derecha
        else:
            esquina = 3  # Cuadrante/ esquina inferior izquierda

        # Si se cumple la distancia minima para esa esquina en específico, hay que actualizar la distancia y la esquina final elegida.
        if match.distance < distanciasMinimas[esquina]:
            distanciasMinimas[esquina] = match.distance
            esquinasFinales[esquina] = punto

    # Si alguno de los cuadrantes no tiene punto asignado, significa que no se detectaron las 4 esquinas.
    if None in esquinasFinales:
        print("No se pudieron detectar las 4 esquinas.")
        exit()

    # Devolver los 4 puntos seleccionados, que deberían corresponder a las esquinas de la hoja.
    return esquinasFinales



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

# Prueba mostrar imagen por pantalla y redimensionarla para que no ocupe todo
def mostrarRedimensionada(titulo, imagen):
    escala=0.2
    alto, ancho = imagen.shape[:2]
    redimensionada = cv2.resize(imagen, (int(ancho * escala), int(alto * escala)))
    cv2.imshow(titulo, redimensionada)

# main
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
        print("Imagen no válida o no cargada")
        exit()

    # Preproceso de la imagen
    imagenPreprocesada = PreprocesarImagen(imagen)

    # Cargar descriptores de referencia
    descriptoresReferencia= CargarDescriptoresReferencia("coordenadas.txt")

    # Detectar esquinas hojas de papel
    esquinas = DetectarEsquinas(imagenPreprocesada, descriptoresReferencia)

    if esquinas is None:
        print(" No se pudieron detectar las 4 esquinas ")
        exit()

    # Rectificar imagen usando las esquinas detectadas
    imagenRectificada = RectificarImagen(imagen, esquinas)

    # Mostrar imágenes para debug y no tener que guardarlas todo el tiempo
    #mostrarRedimensionada('Imagen original', imagen)
    #mostrarRedimensionada('Imagen preprocesada', imagenPreprocesada)
    #mostrarRedimensionada('Imagen rectificada', imagenRectificada)

    # Guardar imagen rectificada
    cv2.imwrite(imagenSalida, imagenRectificada)
    print("Imagen escaneada guardada como:", imagenSalida)

    

