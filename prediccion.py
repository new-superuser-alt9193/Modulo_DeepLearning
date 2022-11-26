import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import PIL

model = tf.keras.models.load_model('modelo_entrenado/emociones_anime_model/')

def predict(model, path):
    class_names = ['Alegria', 'Asco', 'Enojo', 'Miedo', 'Sorpresa', 'Tristeza']
    
    img = tf.keras.utils.load_img(
        path, target_size=(64, 64)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    plt.imshow(img_array[0].numpy().astype("uint8"))

    print(
        "El estado de animo es {} con un {:.2f} porciento de confianza."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

while True:
    print("/////////////////////////////////////////////////////////////////////////////////////")
    print("| Introduzca la direccion de una imagen local, o si desea salir introduzca SALIR    |")
    print("| Ejemplo: /home/alt9193/Documents/IA/Modulo_DeepLearning/data/test/Alegria/111.png |")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print()

    respuesta = input()
    print()
    if respuesta == "SALIR":
        print("/////////////////////////////////////////////////////////////////////////////////////")
        break
    else:
        predict(model, respuesta)
        print()
