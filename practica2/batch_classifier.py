import os
import sys
import cv2
import numpy as np
import joblib

def preprocesar_imagen_rgb(imagen_path, tamaño=(400, 300)):
    img = cv2.imread(imagen_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {imagen_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, tamaño)
    vec = img.flatten().astype(np.float32).reshape(1, -1)
    return vec

def clasificar_imagen(path_imagen, lda, svm_lda, svm_c1, clases):
    vec = preprocesar_imagen_rgb(path_imagen)
    pred_c2 = svm_lda.predict(lda.transform(vec))[0]
    pred_c1 = svm_c1.predict(vec)[0]
    return clases[pred_c2], clases[pred_c1]

if _name_ == "_main_":
    if len(sys.argv) != 2:
        print("Uso: python batch_classifier.py ruta/a/carpeta_con_imagenes")
        sys.exit(1)

    carpeta = sys.argv[1]
    if not os.path.isdir(carpeta):
        print(f"No es una carpeta válida: {carpeta}")
        sys.exit(1)

    try:
        clases = joblib.load("clases.pkl")
        lda = joblib.load("lda_c2.pkl")
        svm_lda = joblib.load("svm_c2.pkl")
        svm_c1 = joblib.load("svm_c1.pkl")
    except Exception as e:
        print(f"Error cargando modelos: {e}")
        sys.exit(1)

    extensiones = ('.png', '.jpg', '.jpeg')

    print(f"Clasificando imágenes en carpeta: {carpeta}\n")

    for root, _, files in os.walk(carpeta):
        for f in files:
            if f.lower().endswith(extensiones):
                ruta_img = os.path.join(root, f)
                try:
                    pred_c2, pred_c1 = clasificar_imagen(ruta_img, lda, svm_lda, svm_c1, clases)
                    print(f"{ruta_img} -> C2: {pred_c2} | C1: {pred_c1}")
                except Exception as e:
                    print(f"Error clasificando {ruta_img}: {e}")