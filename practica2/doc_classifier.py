import sys
import os
import cv2
import numpy as np
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def cargar_imagenes(ruta_base, tamaño=(400, 300)):
    print(f"Cargando imágenes desde: {ruta_base}")
    X, y = [], []
    clases_encontradas = sorted(os.listdir(ruta_base))
    print(f"Clases encontradas: {clases_encontradas}")
    for etiqueta, clase in enumerate(clases_encontradas):
        carpeta = os.path.join(ruta_base, clase)
        if not os.path.isdir(carpeta):
            print(f"No es carpeta: {carpeta}, se omite")
            continue
        archivos = os.listdir(carpeta)
        print(f"Clase '{clase}' - {len(archivos)} archivos")
        for archivo in archivos:
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                ruta_img = os.path.join(carpeta, archivo)
                img = cv2.imread(ruta_img)
                if img is None:
                    print(f" No se pudo cargar: {ruta_img}")
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, tamaño)
                vec = img.flatten().astype(np.float32)
                X.append(vec)
                y.append(etiqueta)
    print(f"Total imágenes cargadas desde {ruta_base}: {len(X)}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), clases_encontradas

def entrenar_solo_svm(X_train, y_train, X_test, y_test):
    print("Entrenando SVM sin reducción (C1)...")
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy C1: {acc:.4f}")
    return svm

def entrenar_lda_svm(X_train, y_train, X_test, y_test, n_componentes=4):
    print("Entrenando LDA + SVM (C2)...")
    lda = LinearDiscriminantAnalysis(n_components=n_componentes)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train_lda, y_train)
    y_pred = svm.predict(X_test_lda)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy C2: {acc:.4f}")
    return lda, svm

def preprocesar_imagen_rgb(imagen_path, tamaño=(400, 300)):
    print(f"Preprocesando imagen: {imagen_path}")
    img = cv2.imread(imagen_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {imagen_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, tamaño)
    vec = img.flatten().astype(np.float32).reshape(1, -1)
    return vec

if __name__ == "__main__":

    try:
        print("Ejecutando doc_classifier.py ...")
        if len(sys.argv) != 2:
            print("Uso incorrecto. Ejecuta como:")
            print("   python doc_classifier.py imagen.jpg")
            sys.exit(1)

        imagen_path = sys.argv[1]
        if not os.path.exists(imagen_path):
            print(f"Imagen no encontrada: {imagen_path}")
            sys.exit(1)

        modelos_existentes = all(os.path.exists(f) for f in ["svm_c1.pkl", "lda_c2.pkl", "svm_c2.pkl", "clases.pkl"])
        print(f"Modelos existentes: {modelos_existentes}")

        if not modelos_existentes:
            print("Modelos no encontrados, entrenando desde cero...")

            ruta_train = 'MUESTRA_PRACTICA2_2025/Aprendizaje'
            ruta_test = 'MUESTRA_PRACTICA2_2025/Test'

            X_train, y_train, clases = cargar_imagenes(ruta_train)
            X_test, y_test, _ = cargar_imagenes(ruta_test)

            svm_c1 = entrenar_solo_svm(X_train, y_train, X_test, y_test)
            joblib.dump(svm_c1, "svm_c1.pkl")
            print("Modelo C1 guardado como svm_c1.pkl")

            lda_c2, svm_c2 = entrenar_lda_svm(X_train, y_train, X_test, y_test)
            joblib.dump(lda_c2, "lda_c2.pkl")
            joblib.dump(svm_c2, "svm_c2.pkl")
            print("Modelos C2 guardados como lda_c2.pkl y svm_c2.pkl")

            joblib.dump(clases, "clases.pkl")
            print("Lista de clases guardada como clases.pkl")

        else:
            print("Modelos encontrados, cargándolos...")

            clases = joblib.load("clases.pkl")
            lda_c2 = joblib.load("lda_c2.pkl")
            svm_c2 = joblib.load("svm_c2.pkl")
            svm_c1 = joblib.load("svm_c1.pkl")

        vec = preprocesar_imagen_rgb(imagen_path)

        vec_reducido = lda_c2.transform(vec)
        pred_c2 = svm_c2.predict(vec_reducido)[0]
        print(f"\nClasificador C2 predice: {clases[pred_c2]}")

        pred_c1 = svm_c1.predict(vec)[0]
        print(f"Clasificador C1 predice: {clases[pred_c1]}")

    except Exception as e:
        print(f"Error inesperado: {e}")