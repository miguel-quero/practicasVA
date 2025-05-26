import sys


import os # Libreria para cargar las imagenes
import cv2
import numpy as np

import joblib # La utilizamos para guardar los modelos creados de la practica.
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from pprint import pprint
import ast  # Para parsear coordenadas desde txt


#                                                                   Funciones para cargar imagenes    
# ------------------------------------------------------------------------------------------------------------------------------------------------------
# Cargar imagenes, vectores y etiquetas de las clases
# Fuente de la funcion flatten: https://stackoverflow.com/questions/46621942/flattening-and-unflattening-an-image
def cargarImagenes(ruta):
    tamaño=(400,300)
    print("Cargando imagenes: ", ruta)

    #Vectores imagenes y etiquetas
    x = []
    y = []
    # Se ordenan las clases alfabéticamente
    clasesEncontradas = sorted(os.listdir(ruta))
    print("Clases: ", clasesEncontradas)

    # Busca las imagenes en las carpetas
    for etiqueta, clase in enumerate(clasesEncontradas):
        carpeta = os.path.join(ruta, clase)

        for archivo in os.listdir(carpeta):
            if archivo.lower().endswith(('.png','.jpg','.jpeg')):

                rutaImagen = os.path.join(carpeta, archivo)
                img = cv2.imread(rutaImagen)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Se cambia el tamaño de la imagen
                img = cv2.resize(img, tamaño)
                # vector de dimensión 1 para SVM y LDA
                vector = img.flatten().astype(np.float32)

                x.append(vector)
                y.append(etiqueta)

    # Hay que convertir las listas a arrays de numpy para no producir error ejecucion
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.int32), clasesEncontradas



# Funcion que carga las imagenes igual que la anterior, pero ya rectificadas
# Fuente de la funcion flatten: https://stackoverflow.com/questions/46621942/flattening-and-unflattening-an-image
def cargarImagenesRectificadas(ruta, rutaCoordenadas):
    print("Cargando imagenes rectificadas: ", ruta)
    tamaño=(400, 300)
    #Vectores imagenes y etiquetas
    x = []
    y = []

    # Se ordenan las clases alfabéticamente
    clasesEncontradas = sorted(os.listdir(ruta))
    
    # Busca las imagenes en las carpetas
    for etiqueta, clase in enumerate(clasesEncontradas):
        carpeta = os.path.join(ruta, clase)

        for archivo in os.listdir(carpeta):
            if archivo.lower().endswith(('.png', '.jpg', '.jpeg')):
                
                # Conseguir las esquinas de la imagen del archivo txt
                esquinas = rutaCoordenadas[archivo]

                rutaImagen = os.path.join(carpeta, archivo)
                img = cv2.imread(rutaImagen)

                 # Se rectifica la imagen con nuestra funcion de la practica 1
                img = RectificarImagen(img, esquinas)
                 # Se cambia el tamaño de la imagen
                img = cv2.resize(img, tamaño)

                # vector de dimensión 1 para SVM y LDA
                vector = img.flatten().astype(np.float32)

                x.append(vector)
                y.append(etiqueta)

    # Hay que convertir las listas a arrays de numpy para no producir error ejecucion
    return np.array(x, dtype=np.float32), np.array(y, dtype=np.int32), clasesEncontradas





# --------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                   Entrenamiento de modelos


# Clasificador 1: SVM
def EntrenamientoSVMC1(xTrain, yTrain, xTest, yTest):
    print("Entrenamiento SVM (C1)")
    # Crear clasificador SVM
    svm =SVC(kernel='linear',random_state=42)

    #Entrenamiento con las muestras de entrenamiento
    svm.fit(xTrain,yTrain)

    # Clasificación de las muestras de test
    yPredict =svm.predict(xTest)

    # Métrica de precision
    accuracy =accuracy_score(yTest,yPredict)
    print("Accuracy C1: "+str(round(accuracy,4)))

    # Se devuelve el clasificador entrenado
    return svm, accuracy

# Clasificador 2: LDA + SVM
def EntrenamientoLDASVMC2(xTrain, yTrain, xTest, yTest):
    print("Entrenamiento LDA + SVM (C2)")

    # Conseguir numero de clases para el LDA
    numClases =len(np.unique(yTrain))

    # Crear el LDA
    lda= LinearDiscriminantAnalysis(n_components=numClases - 1)
    # Aplicar el LDA a las muestras de entrenamiento y Test
    xTrainLDA= lda.fit_transform(xTrain,yTrain)
    xTestLDA = lda.transform(xTest)

    # Crear clasificador SVM
    svm = SVC(kernel='linear',random_state=42)

    #Entrenamiento con las muestras de entrenamiento del LDA
    svm.fit(xTrainLDA, yTrain)

    # Clasificación de las muestras de test del LDA
    yPredict = svm.predict(xTestLDA)

    # Métrica de precision
    accuracy = accuracy_score(yTest, yPredict)
    print("Accuracy C2: "+ str(round(accuracy,4)))

    # Se devuelven el SVM y el LDA ya entrenados
    return lda, svm

# Clasificador 3: SVM con imagenes rectificadas
def EntrenamientoSVMC3(xTrain, yTrain,xTest,yTest):
    print("Entrenamiento SVM con imagenes rectificadas (C3)")
    # Crear clasificador SVM
    svm =SVC(kernel='linear',random_state=42)

    #Entrenamiento con las muestras de entrenamiento
    svm.fit(xTrain,yTrain)

    # Clasificación de las muestras de test
    yPredict = svm.predict(xTest)

    # Métrica de precision
    accuracy =accuracy_score(yTest,yPredict)
    print("Accuracy C3: "+ str(round(accuracy,4)))

    # Se devuelve el clasificador SVM entrenado
    return svm


# --------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                   Funciones Auxiliares de preproceso, rectificación y lectura de coordenadas

# Funcion para rectificar imagenes de nuestra practica 1
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


# Leer las esquinas de las coordenadas de cada imagen para guardarlas en el diccionario y así cargar las luego más facilmente 
# Fuente de la funcion ast.literal_eval: https://docs.python.org/3/library/ast.html#ast.literal_eval
def LeerEsquinas(ruta):

    # Diccionario para guardar las esquinas
    esquinas = {}

    # Se abre el archivo para leer
    with open(ruta, 'r') as f:
        for linea in f:
            # Ignorar las lineas vacías
            if ':' not in linea:
                continue
            # separación del nombre de la imagen y sus coordenadas de esquina
            nombre, coordenadas = linea.split(':', 1)

            # Se utiliza para parsear las coordenadas de las esquinas
            coords = ast.literal_eval(coordenadas.strip())

            # Aqui se guardan las coordenadas en el diccionario
            esquinas[nombre.strip()] = coords

    # Se devuelve el diccionario de esquinas        
    return esquinas


# Preprocesa una imagen RGB para los clasificadores C1 y C2.
# Aplica conversión a RGB, redimensionamiento, aplanado (flatten) y normalizacion con el scaler correspondiente.
def preprocesar_imagen_rgb (rutaImagen):
    tamaño=(400,300)
    img= cv2.imread(rutaImagen)

    # Convertimos de BGR (formato por defecto de OpenCV) a RGB
    img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Redimensionamos la imagen al tamaño
    img = cv2.resize(img,tamaño)

    # Aplanamos la imagen (vector de dimension 1) y convertimos a tipo float32
    vector =img.flatten().astype(np.float32).reshape(1, -1)

    # Cargamos el scaler ya entrenado para normalizar los vectores
    scaler =joblib.load("scaler.pkl")

    # Se devuelve el vector ya normalizado directamente
    return scaler.transform(vector)



# Preprocesa una imagen rectificada para el clasificador C3.
# Aplica la rectificación usando la función RectificarImagen, luego resize, flatten y normalización con el scaler (normalizador) del clasificador 3.
def preprocesar_imagen_c3(rutaImagen, esquinas):
    tamaño=(400, 300)
    img =cv2.imread(rutaImagen)

    # Obtener el nombre del archivo para buscar sus esquinas
    nombreImagen = os.path.basename(rutaImagen)

    # Rectificar la imagen con las esquinas usando la función definida anteriormente
    img= RectificarImagen(img, esquinas[nombreImagen])

    # Redimensionar y convertir a vector de 1 dimensión
    img = cv2.resize(img, tamaño)
    vector =img.flatten().astype(np.float32).reshape(1, -1)

    # Normalizar con el scaler correspondiente al clasificador C3
    scaler_c3 = joblib.load("scaler_c3.pkl")

    # Se devuelve el vector ya normalizado directamente
    return scaler_c3.transform(vector)


# Preprocesa una imagen rectificada para el clasificador C4 (LDA + SVM sobre imágenes rectificadas).
# Aplica la rectificación con RectificarImagen, luego resize, flatten, normalización con scaler_c3 y transformación de las muestras con lda_c4.
def preprocesar_imagen_rectificada_lda_c4(rutaImagen, esquinas):
    tamaño=(400, 300)
    img = cv2.imread(rutaImagen)

    # Obtener el nombre del archivo (imagen) para recuperar sus coordenadas
    nombreImagen =os.path.basename(rutaImagen)

    # Rectificar la imagen aplicando la transformacion de perspectiva
    img= RectificarImagen(img, esquinas[nombreImagen])

    # Redimensionar y aplanar la imagen
    img = cv2.resize(img,tamaño)
    vector = img.flatten().astype(np.float32).reshape(1,-1)

    # Normalizar con el scaler de C3
    scaler_c3 =joblib.load("scaler_c3.pkl")
    vectorC3 = scaler_c3.transform(vector)

    # Aplicar reducción de dimensionalidad con LDA entrenado
    lda_c4= joblib.load("lda_c4.pkl")

    # Se devuelve el vector con la transformacion del LDA
    return lda_c4.transform(vectorC3)


# --------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                   Programa Principal
# main
# Fuente de la libreria joblib (para el guardado de modelos): https://scikit-learn.org/stable/model_persistence.html
# Fuentes del normalizador de características que hemos utilizado para mejorar el SVM y el LDA
# https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling
# https://stackoverflow.com/questions/14688391/how-to-apply-standardization-to-svms-in-scikit-learn
# Hemos utilizado StandardScaler porque normaliza los datos a la distribución normal estándar N(0,1) para mejorar las metricas de los clasificadores.
if __name__ == "__main__":
    try:
        print("Ejecutando doc_classifier.py ...")

        # Comprobar que se ha pasado correctamente el nombre de la imagen a clasificar como argumento
        if len(sys.argv) != 2:
            print("Uso incorrecto. Ejecuta como:\n   python doc_classifier.py imagen.jpg")
            sys.exit(1)

        # Obtener la ruta de la imagen a clasificar
        rutaImagen = sys.argv[1]
        if not os.path.exists(rutaImagen):
            print(f"Imagen no encontrada: {rutaImagen}")
            sys.exit(1)

        # Comprobar si ya existen los modelos previamente entrenados
        modelos_existentes = all(os.path.exists(f) for f in [
            "svm_c1.pkl", "lda_c2.pkl", "svm_c2.pkl", "clases.pkl", "scaler.pkl",
            "svm_c3.pkl", "scaler_c3.pkl", "clases_c3.pkl", "coordenadasprac2.txt"
        ])
        print(f"Modelos existentes: {modelos_existentes}")

        # Si no existen los modelos, se realiza el entrenamiento desde cero
        if not modelos_existentes:
            print("Modelos no encontrados, entrenando desde cero...")

            # Rutas de las carpetas de entrenamiento y test, y del archivo de coordenadas
            ruta_train = 'MUESTRA_PRACTICA2_2025/Aprendizaje'
            ruta_test = 'MUESTRA_PRACTICA2_2025/Test'
            txt_coordenadas = 'coordenadasprac2.txt'

            # Cargar imágenes RGB y etiquetas para entrenamiento y test (clasificadores C1 y C2)
            X_train, y_train, clases = cargarImagenes(ruta_train)
            X_test, y_test, _ = cargarImagenes(ruta_test)

            # Normalizar vectores de características
            print("Normalizando características...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            joblib.dump(scaler, "scaler.pkl")

            # Entrenamiento del clasificador C1: SVM sobre imágenes RGB
            svm_c1, acc_c1 = EntrenamientoSVMC1(X_train_scaled, y_train, X_test_scaled, y_test)
            joblib.dump(svm_c1, "svm_c1.pkl")
            joblib.dump(clases, "clases.pkl")



            # Entrenamiento del clasificador C2: LDA + SVM sobre imágenes RGB
            lda_c2, svm_c2 = EntrenamientoLDASVMC2(X_train_scaled, y_train, X_test_scaled, y_test)
            joblib.dump(lda_c2, "lda_c2.pkl")
            joblib.dump(svm_c2, "svm_c2.pkl")

            # Leer coordenadas de las esquinas de todas las imágenes desde archivo
            esquinas = LeerEsquinas(txt_coordenadas)

            # Cargar imágenes rectificadas para entrenamiento y test (clasificadores C3 y C4)
            X_train_c3, y_train_c3, clases_c3 = cargarImagenesRectificadas(ruta_train, esquinas)
            X_test_c3, y_test_c3, _ = cargarImagenesRectificadas(ruta_test, esquinas)

            # Verificar que haya datos suficientes para entrenar con imágenes rectificadas
            if X_train_c3.size == 0 or X_test_c3.size == 0:
                raise ValueError("No hay datos para entrenar o evaluar el clasificador C3 con imágenes rectificadas.")

            # Normalizar vectores de características de imágenes rectificadas
            print("Normalizando características rectificadas...")
            scaler_c3 = StandardScaler()
            X_train_c3_scaled = scaler_c3.fit_transform(X_train_c3)
            X_test_c3_scaled = scaler_c3.transform(X_test_c3)
            joblib.dump(scaler_c3, "scaler_c3.pkl")

            # Entrenamiento del clasificador C3: SVM sobre imágenes rectificadas
            svm_c3 = EntrenamientoSVMC3(X_train_c3_scaled, y_train_c3, X_test_c3_scaled, y_test_c3)
            joblib.dump(svm_c3, "svm_c3.pkl")
            joblib.dump(clases_c3, "clases_c3.pkl")

            # Entrenamiento del clasificador C4: LDA + SVM sobre imágenes rectificadas
            lda_c4, svm_c4 = EntrenamientoLDASVMC2(X_train_c3_scaled, y_train_c3, X_test_c3_scaled, y_test_c3)
            y_pred_c4 = svm_c4.predict(lda_c4.transform(X_test_c3_scaled))
            acc_c4 = accuracy_score(y_test_c3, y_pred_c4)
            print(f"Accuracy C4: {acc_c4:.4f}")

            # Guardar los modelos entrenados
            joblib.dump(lda_c4, "lda_c4.pkl")
            joblib.dump(svm_c4, "svm_c4.pkl")

            print("Modelos y etiquetas guardados correctamente.")

        # Si ya existen los modelos, simplemente se cargan desde los archivos .pkl
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
            esquinas = LeerEsquinas("coordenadasprac2.txt")
            ruta_test = 'MUESTRA_PRACTICA2_2025/Test'
            X_test, y_test, _ = cargarImagenes(ruta_test)
            X_test_scaled = scaler.transform(X_test)

            esquinas = LeerEsquinas("coordenadasprac2.txt")
            X_test_c3, y_test_c3, _ = cargarImagenesRectificadas(ruta_test, esquinas)
            X_test_c3_scaled = scaler_c3.transform(X_test_c3)
            acc_c1 = accuracy_score(y_test, svm_c1.predict(X_test_scaled))
            acc_c4 = accuracy_score(y_test_c3, svm_c4.predict(lda_c4.transform(X_test_c3_scaled)))

            


        # Preprocesar la imagen pasada por argumento para cada uno de los clasificadores
        vector = preprocesar_imagen_rgb(rutaImagen)
        vec_lda = lda_c2.transform(vector)
        pred_c2 = svm_c2.predict(vec_lda)[0]
        pred_c1 = svm_c1.predict(vector)[0]

        vec_c3 = preprocesar_imagen_c3(rutaImagen, esquinas)
        pred_c3 = svm_c3.predict(vec_c3)[0]

        vec_c4 = preprocesar_imagen_rectificada_lda_c4(rutaImagen, esquinas)
        pred_c4 = svm_c4.predict(vec_c4)[0]

        acc_c2 = accuracy_score(y_test, svm_c2.predict(lda_c2.transform(X_test_scaled)))
        acc_c3 = accuracy_score(y_test_c3, svm_c3.predict(X_test_c3_scaled))


        # Mostrar predicción de cada clasificador por pantalla
        print("\n" + "-"*50)
        print("RESULTADO DE CLASIFICACIÓN")
        print("-"*50)
        print(f"{'Clasificador':<25} {'Predicción':<20}")
        print("-"*50)
        print(f"{'C1 - SVM sobre RGB':<25} {clases[pred_c1]:<20}")
        print(f"{'C2 - LDA + SVM sobre RGB':<25} {clases[pred_c2]:<20}")
        print(f"{'C3 - SVM sobre rectificada':<25} {clases_c3[pred_c3]:<20}")
        print(f"{'C4 - LDA + SVM rectificada':<25} {clases_c3[pred_c4]:<20}")
        print("-"*50)

        # Tabla de comparación de accuracy por clasificador
        print("\n" + "="*60)
        print("COMPARACIÓN DE LOS MODELOS (ACCURACY)")
        print("="*60)
        print(f"{'Clasificador':<25} {'Entrada':<20} {'Accuracy':<10}")
        print("-"*60)
        print(f"{'C1 - SVM':<25} {'Imagen RGB':<20} {round(acc_c1, 4):<10}")
        print(f"{'C2 - LDA + SVM':<25} {'Imagen RGB':<20} {round(accuracy_score(y_test, svm_c2.predict(lda_c2.transform(X_test_scaled))), 4):<10}")
        print(f"{'C3 - SVM':<25} {'Imagen rectificada':<20} {round(accuracy_score(y_test_c3, svm_c3.predict(X_test_c3_scaled)), 4):<10}")
        print(f"{'C4 - LDA + SVM':<25} {'Imagen rectificada':<20} {round(acc_c4, 4):<10}")
        print("="*60)



    # Capturar errores inesperados y mostrarlos por pantalla
    except Exception as e:
        print(f"Error inesperado: {e}")
