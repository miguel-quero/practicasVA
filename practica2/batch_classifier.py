import os
import sys
import cv2
import numpy as np
import joblib
import ast

def leer_esquinas(txt_path):
    esquinas_dict = {}
    with open(txt_path, 'r') as f:
        for linea in f:
            if ':' not in linea:
                continue
            nombre, coords_str = linea.split(':', 1)
            coords = ast.literal_eval(coords_str.strip())
            esquinas_dict[nombre.strip()] = coords
    return esquinas_dict

def RectificarImagen(imagen, esquinas):
    srcImagen = np.array(esquinas, np.float32)
    ancho = int(np.linalg.norm(srcImagen[0] - srcImagen[1]))
    alto = int(ancho * np.sqrt(2))
    dstImagen = np.array([
        [0, 0],
        [ancho, 0],
        [ancho, alto],
        [0, alto]
    ], np.float32)
    matrizPerspectiva = cv2.getPerspectiveTransform(srcImagen, dstImagen)
    return cv2.warpPerspective(imagen, matrizPerspectiva, (ancho, alto))

def preprocesar_imagen_rgb(imagen_path, tamaño=(400, 300)):
    img = cv2.imread(imagen_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {imagen_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, tamaño)
    vec = img.flatten().astype(np.float32).reshape(1, -1)
    return vec

def preprocesar_imagen_c3(imagen_path, esquinas_dict, tamaño=(400, 300)):
    img = cv2.imread(imagen_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {imagen_path}")
    base_name = os.path.basename(imagen_path)
    if base_name not in esquinas_dict:
        raise ValueError(f"Coordenadas no encontradas para {base_name}")
    img_rect = RectificarImagen(img, esquinas_dict[base_name])
    # Mantener imagen en color para que coincida con entrenamiento
    img_resized = cv2.resize(img_rect, tamaño)
    vec = img_resized.flatten().astype(np.float32).reshape(1, -1)
    return vec

def clasificar_imagen(path_imagen, lda, svm_lda, svm_c1, svm_c3, scaler, scaler_c3, clases, clases_c3, esquinas_dict):
    # Preprocesar para C1 y C2
    vec = preprocesar_imagen_rgb(path_imagen)
    vec_scaled = scaler.transform(vec)
    pred_c2 = svm_lda.predict(lda.transform(vec_scaled))[0]
    pred_c1 = svm_c1.predict(vec_scaled)[0]

    # Preprocesar para C3 (imagen rectificada)
    try:
        vec_c3 = preprocesar_imagen_c3(path_imagen, esquinas_dict)
        vec_c3_scaled = scaler_c3.transform(vec_c3)
        pred_c3 = svm_c3.predict(vec_c3_scaled)[0]
        pred_c3_label = clases_c3[pred_c3]
    except Exception as e:
        pred_c3_label = f"Error C3: {e}"

    return clases[pred_c2], clases[pred_c1], pred_c3_label

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python batch_classifier.py carpeta coordenadas.txt")
        sys.exit(1)

    carpeta = sys.argv[1]
    archivo_coordenadas = sys.argv[2]

    if not os.path.isdir(carpeta):
        print(f"No es una carpeta válida: {carpeta}")
        sys.exit(1)
    if not os.path.exists(archivo_coordenadas):
        print(f"No se encontró el archivo de coordenadas: {archivo_coordenadas}")
        sys.exit(1)

    try:
        clases = joblib.load("clases.pkl")
        lda = joblib.load("lda_c2.pkl")
        svm_lda = joblib.load("svm_c2.pkl")
        svm_c1 = joblib.load("svm_c1.pkl")
        scaler = joblib.load("scaler.pkl")

        svm_c3 = joblib.load("svm_c3.pkl")
        scaler_c3 = joblib.load("scaler_c3.pkl")
        clases_c3 = joblib.load("clases_c3.pkl")

        esquinas_dict = leer_esquinas(archivo_coordenadas)

    except Exception as e:
        print(f"Error cargando modelos o coordenadas: {e}")
        sys.exit(1)

    extensiones = ('.png', '.jpg', '.jpeg')
    print(f"\nClasificando imágenes en carpeta: {carpeta}\n")

    total = 0
    correctos_c1 = 0
    correctos_c2 = 0
    correctos_c3 = 0

    for root, _, files in os.walk(carpeta):
        clase_real = os.path.basename(root)
        for f in files:
            if f.lower().endswith(extensiones):
                ruta_img = os.path.join(root, f)
                try:
                    pred_c2, pred_c1, pred_c3 = clasificar_imagen(ruta_img, lda, svm_lda, svm_c1, svm_c3, scaler, scaler_c3, clases, clases_c3, esquinas_dict)
                    print(f"{ruta_img} -> C2: {pred_c2} | C1: {pred_c1} | C3: {pred_c3}")
                    total += 1
                    if pred_c1.lower() == clase_real.lower():
                        correctos_c1 += 1
                    if pred_c2.lower() == clase_real.lower():
                        correctos_c2 += 1
                    if pred_c3.lower() == clase_real.lower():
                        correctos_c3 += 1
                except Exception as e:
                    print(f"Error clasificando {ruta_img}: {e}")

    print("\nRESULTADOS FINALES:")
    if total > 0:
        acc_c1 = 100 * correctos_c1 / total
        acc_c2 = 100 * correctos_c2 / total
        acc_c3 = 100 * correctos_c3 / total
        print(f"Clasificador C1 acierta {correctos_c1}/{total} imágenes ({acc_c1:.2f}%)")
        print(f"Clasificador C2 acierta {correctos_c2}/{total} imágenes ({acc_c2:.2f}%)")
        print(f"Clasificador C3 acierta {correctos_c3}/{total} imágenes ({acc_c3:.2f}%)")
    else:
        print("No se encontraron imágenes para evaluar.")
