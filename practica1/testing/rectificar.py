import cv2
import numpy as np

# Diccionario de coordenadas por imagen
coordenadas = {
    "Hoja_01.jpg": [(397, 280), (2649, 481), (2674, 3507), (401, 3695)],
    "Hoja_15.jpg": [(538, 602), (2494, 585), (2691, 3494), (466, 3528)],
    "Hoja_06.jpg": [(261, 491), (2618, 457), (2776, 3729), (427, 3908)],
    "Hoja_07.jpg": [(483, 367), (2511, 487), (2605, 3396), (337, 3455)],
    "Hoja_08.jpg": [(500, 491), (2524, 551), (2793, 3613), (248, 3673)],
    "Hoja_09.jpg": [(718, 333), (2580, 581), (2849, 3400), (367, 3361)],
    "Hoja_10.jpg": [(721, 283), (2741, 346), (2706, 3363), (431, 3172)],
    "Hoja_11.jpg": [(436, 517), (2520, 551), (2746, 3643), (376, 3626)],
    "Hoja_12.jpg": [(534, 658), (2460, 645), (2482, 3310), (594, 3370)],
    "Hoja_13.jpg": [(380, 359), (2601, 466), (2601, 3549), (342, 3613)],
    "Hoja_14.jpg": [(316, 619), (2379, 491), (2704, 3481), (525, 3643)],
    "Hoja_02.jpg": [(295, 414), (2973, 414), (2652, 3904), (427, 3677)],
    "Hoja_03.jpg": [(401, 329), (2576, 431), (2576, 3554), (316, 3519)],
    "Hoja_04.jpg": [(299, 431), (2529, 483), (2571, 3468), (393, 3584)],
    "Hoja_05.jpg": [(572, 333), (2674, 666), (2687, 3870), (81, 3673)]
}
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


for nombre, puntos in coordenadas.items():
    imagen = cv2.imread(nombre)
    if imagen is None:
        print(f"No se pudo cargar la imagen: {nombre}")
        continue
    imagen_rectificada = RectificarImagen(imagen, puntos)
    salida = f"rectificada_{nombre}"
    cv2.imwrite(salida, imagen_rectificada)
    print(f"Imagen guardada: {salida}")