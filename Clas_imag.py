'''
Clasificador de imágenes de logotipos, empleando un modelo de redes convolucionales previamente entrenado
'''
if __name__ == "__main__":
    
    # Importamos las librerías necesarias
    #
    import numpy as np
    import pandas as pd
    import os
    import cv2                                 
    import tensorflow as tf                
    from tqdm import tqdm

    #Asignamos las variables necesarias
    IMAGE_SIZE = (128, 128)
    class_names = ['no_nike', 'nike']

    #Cambiamos nuestra ruta al directorio actual
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    #Función para cargar las imágenes
    def load_data():

        dataset = "Test_Imag"
        output = []
        images = []
        labels = []
        
        # Recorremos el directorio
        for file in os.listdir(dataset):
            # Obtenemos la dirección de la imagen
            img_path = os.path.join(dataset, file)
                
            # Abrimos y modificamos el tamaño de la imagen
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, IMAGE_SIZE) 
                
            # Añadimos la imagen a nuestro conjunto
            images.append(image)
                    
        images = np.array(images, dtype = 'float32')
        return images

    loaded_model = tf.keras.models.load_model("clasificador_nike.h5")   #Cargamos nuestro modelo preentrenado

    pred_images = load_data()   #Leemos nuestras imágenes
    pred_images = pred_images/255.0   #Normalizamos las imágenes

    prediction = loaded_model.predict(pred_images)   # Realizamos nuestra predicción

    lista_pred = []   # Variable (lista) en la que almacenaré mis predicciones
    for pred in prediction:
        if pred >= 0.6:
            lista_pred.append(True)
        else:
            lista_pred.append(False)   


    print(lista_pred)



        