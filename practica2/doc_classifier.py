import sys
import os
import cv2
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from pprint import pprint
import ast  # Para parsear coordenadas desde txt

def cargar_imagenes(ruta_base, tamaño=(400, 300)):
    print(f"Cargando imágenes desde: {ruta_base}")
    X, y = [], []
    clases_encontradas = sorted(os.listdir(ruta_base))
    print(f"Clases encontradas: {clases_encontradas}")
    for etiqueta, clase in enumerate(clases_encontradas):
        carpeta = os.path.join(ruta_base, clase)
        if not os.path.isdir(carpeta):
            continue
        for archivo in os.listdir(carpeta):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                ruta_img = os.path.join(carpeta, archivo)
                img = cv2.imread(ruta_img)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, tamaño)
                vec = img.flatten().astype(np.float32)
                X.append(vec)
                y.append(etiqueta)
    print(f"Total imágenes cargadas: {len(X)}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), clases_encontradas

def cargar_imagenes_rectificadas(ruta_base, esquinas_dict, tamaño=(400, 300)):
    X, y = [], []
    clases_encontradas = sorted(os.listdir(ruta_base))
    print(f"Cargando imágenes rectificadas desde: {ruta_base}")
    for etiqueta, clase in enumerate(clases_encontradas):
        carpeta = os.path.join(ruta_base, clase)
        if not os.path.isdir(carpeta):
            continue
        for archivo in os.listdir(carpeta):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                if archivo not in esquinas_dict:
                    print(f"Advertencia: no se encontró esquinas para {archivo}, se omite")
                    continue
                esquinas = esquinas_dict[archivo]
                if len(esquinas) != 4:
                    print(f"Advertencia: esquinas inválidas para {archivo}, se omite")
                    continue
                ruta_img = os.path.join(carpeta, archivo)
                img = cv2.imread(ruta_img)
                if img is None:
                    continue
                img_rect = RectificarImagen(img, esquinas)
                img_resized = cv2.resize(img_rect, tamaño)
                vec = img_resized.flatten().astype(np.float32)
                X.append(vec)
                y.append(etiqueta)
    print(f"Total imágenes rectificadas cargadas desde {ruta_base}: {len(X)}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), clases_encontradas



def entrenar_solo_svm(X_train, y_train, X_test, y_test):
    pprint("Entrenando SVM sin reducción (C1)...")
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy C1: {acc:.4f}")
    return svm

def entrenar_lda_svm(X_train, y_train, X_test, y_test):
    print("Entrenando LDA + SVM (C2)...")
    n_clases = len(np.unique(y_train))
    lda = LinearDiscriminantAnalysis(n_components=n_clases - 1)
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train_lda, y_train)
    y_pred = svm.predict(X_test_lda)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy C2: {acc:.4f}")
    return lda, svm

def entrenar_svm(X_train, y_train, X_test, y_test):
    pprint("Entrenando SVM sin reducción (C3)...")
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy C3: {acc:.4f}")
    return svm

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

def preprocesar_imagen_rgb(imagen_path, tamaño=(400, 300)):
    img = cv2.imread(imagen_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {imagen_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, tamaño)
    vec = img.flatten().astype(np.float32).reshape(1, -1)
    scaler = joblib.load("scaler.pkl")
    return scaler.transform(vec)

def preprocesar_imagen_c3(imagen_path, esquinas_dict, tamaño=(400, 300)):
    img = cv2.imread(imagen_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {imagen_path}")
    base_name = os.path.basename(imagen_path)
    if base_name not in esquinas_dict:
        raise ValueError(f"Coordenadas no encontradas para {base_name}")
    img_rect = RectificarImagen(img, esquinas_dict[base_name])
    img_resized = cv2.resize(img_rect, tamaño)
    vec = img_resized.flatten().astype(np.float32).reshape(1, -1)
    scaler_c3 = joblib.load("scaler_c3.pkl")
    return scaler_c3.transform(vec)

def preprocesar_imagen_rectificada_lda_c4(imagen_path, esquinas_dict, tamaño=(400, 300)):
    img = cv2.imread(imagen_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {imagen_path}")
    base_name = os.path.basename(imagen_path)
    if base_name not in esquinas_dict:
        raise ValueError(f"Coordenadas no encontradas para {base_name}")
    img_rect = RectificarImagen(img, esquinas_dict[base_name])
    img_resized = cv2.resize(img_rect, tamaño)
    vec = img_resized.flatten().astype(np.float32).reshape(1, -1)
    scaler_c3 = joblib.load("scaler_c3.pkl")
    lda_c4 = joblib.load("lda_c4.pkl")
    vec_scaled = scaler_c3.transform(vec)
    return lda_c4.transform(vec_scaled)


if __name__ == "__main__":
    try:
        print("Ejecutando doc_classifier.py ...")
        if len(sys.argv) != 2:
            print("Uso incorrecto. Ejecuta como:\n   python doc_classifier.py imagen.jpg")
            sys.exit(1)

        imagen_path = sys.argv[1]
        if not os.path.exists(imagen_path):
            print(f"Imagen no encontrada: {imagen_path}")
            sys.exit(1)

        modelos_existentes = all(os.path.exists(f) for f in [
            "svm_c1.pkl", "lda_c2.pkl", "svm_c2.pkl", "clases.pkl", "scaler.pkl",
            "svm_c3.pkl", "scaler_c3.pkl", "clases_c3.pkl", "coordenadasprac2.txt"
        ])
        print(f"Modelos existentes: {modelos_existentes}")

        if not modelos_existentes:
            print("Modelos no encontrados, entrenando desde cero...")

            ruta_train = 'MUESTRA_PRACTICA2_2025/Aprendizaje'
            ruta_test = 'MUESTRA_PRACTICA2_2025/Test'
            txt_coordenadas = 'coordenadasprac2.txt'

            X_train, y_train, clases = cargar_imagenes(ruta_train)
            X_test, y_test, _ = cargar_imagenes(ruta_test)

            print("Normalizando características...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            joblib.dump(scaler, "scaler.pkl")

            svm_c1 = entrenar_solo_svm(X_train_scaled, y_train, X_test_scaled, y_test)
            joblib.dump(svm_c1, "svm_c1.pkl")

            lda_c2, svm_c2 = entrenar_lda_svm(X_train_scaled, y_train, X_test_scaled, y_test)
            joblib.dump(lda_c2, "lda_c2.pkl")
            joblib.dump(svm_c2, "svm_c2.pkl")

            # Leer coordenadas de todas las imágenes (train y test)
            esquinas_dict = leer_esquinas(txt_coordenadas)

            # Cargar imágenes rectificadas para entrenamiento y test usando coordenadas
            X_train_c3, y_train_c3, clases_c3 = cargar_imagenes_rectificadas(ruta_train, esquinas_dict)
            X_test_c3, y_test_c3, _ = cargar_imagenes_rectificadas(ruta_test, esquinas_dict)

            if X_train_c3.size == 0 or X_test_c3.size == 0:
                raise ValueError("No hay datos para entrenar o evaluar el clasificador C3 con imágenes rectificadas.")

            print("Normalizando características rectificadas...")
            scaler_c3 = StandardScaler()
            X_train_c3_scaled = scaler_c3.fit_transform(X_train_c3)
            X_test_c3_scaled = scaler_c3.transform(X_test_c3)
            joblib.dump(scaler_c3, "scaler_c3.pkl")

            svm_c3 = entrenar_svm(X_train_c3_scaled, y_train_c3, X_test_c3_scaled, y_test_c3)
            joblib.dump(svm_c3, "svm_c3.pkl")
            joblib.dump(clases_c3, "clases_c3.pkl")

            # Utilizar funcion reducción funcionalidad (entrenar_lda_svm) 3.2
            lda_c4, svm_c4 = entrenar_lda_svm(X_train_c3_scaled, y_train_c3, X_test_c3_scaled, y_test_c3)
            y_pred_c4 = svm_c4.predict(lda_c4.transform(X_test_c3_scaled))
            acc_c4 = accuracy_score(y_test_c3, y_pred_c4)
            print(f"Accuracy C4: {acc_c4:.4f}")

            joblib.dump(lda_c4, "lda_c4.pkl")
            joblib.dump(svm_c4, "svm_c4.pkl")

            print("Modelos y etiquetas guardados correctamente.")

        else:
            print("Modelos encontrados, cargándolos...")
            clases = joblib.load("clases.pkl")
            scaler = joblib.load("scaler.pkl")
            lda_c2 = joblib.load("lda_c2.pkl")
            svm_c2 = joblib.load("svm_c2.pkl")
            svm_c1 = joblib.load("svm_c1.pkl")

            clases_c3 = joblib.load("clases_c3.pkl")
            scaler_c3 = joblib.load("scaler_c3.pkl")
            svm_c3 = joblib.load("svm_c3.pkl")
            lda_c4 = joblib.load("lda_c4.pkl")
            svm_c4 = joblib.load("svm_c4.pkl")
            esquinas_dict = leer_esquinas("coordenadasprac2.txt")

        vec = preprocesar_imagen_rgb(imagen_path)
        vec_lda = lda_c2.transform(vec)
        pred_c2 = svm_c2.predict(vec_lda)[0]
        pred_c1 = svm_c1.predict(vec)[0]

        vec_c3 = preprocesar_imagen_c3(imagen_path, esquinas_dict)
        pred_c3 = svm_c3.predict(vec_c3)[0]

        vec_c4 = preprocesar_imagen_rectificada_lda_c4(imagen_path, esquinas_dict)
        pred_c4 = svm_c4.predict(vec_c4)[0]

        print(f"\nClasificador C1 predice: {clases[pred_c1]}")
        print(f"Clasificador C2 predice: {clases[pred_c2]}")
        print(f"Clasificador C3 predice: {clases_c3[pred_c3]}")
        print(f"Clasificador C4 predice: {clases_c3[pred_c4]}")

    except Exception as e:
        print(f"Error inesperado: {e}")
