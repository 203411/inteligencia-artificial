from matplotlib.cbook import flatten
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras.models import Sequential
import seaborn as sns
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier



batch_size = 1000
img_height = 100
img_width = 100

def cargar_imagenes():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        "./data/Entrenamiento",
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        "./data/Validacion",
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names
    plt.figure(figsize=(100, 100))

    for images, labels in train_ds.take(1):
        for i in range(100):
            ax = plt.subplot(10, 10, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()

    train_ds = train_ds.cache().prefetch(buffer_size=10)
    val_ds = val_ds.cache().prefetch(buffer_size=10)
    num_clases = len(class_names)

    return val_ds,train_ds,class_names

def rotacion_imagenes(train_ds):
    data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal",
                        input_shape=(img_height,
                                    img_width,
                                    3)),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ]
    )

    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            ax = plt.subplot(4, 4, i + 1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")
    plt.show()

def funcion_activacion ():
        modelo = Sequential([
        tf.keras.layers.Flatten(input_shape=(img_height, img_width,3)),
        tf.keras.layers.Dense(100,activation=tf.nn.relu),
        tf.keras.layers.Dense(50,activation=tf.nn.relu),
        tf.keras.layers.Dense(20 ,activation=tf.nn.softmax),
    
        ])
        modelo.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy'])
        
        return modelo

def entrenamiento(modelo):
        epochs = 100
        history = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs)
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)
        modelo.save('modelo.h5')
        modelo_json = modelo.to_json()
        with open('modelo.json', 'w') as archivo_json:
            archivo_json.write(modelo_json)
        return epochs_range ,acc,val_acc,loss,val_loss

def graficas(epochs_range,acc,val_acc,loss,val_loss):
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Precision de entrenamiento')
    plt.plot(epochs_range, val_acc, label='Precision de validación')
    plt.legend(loc='lower right')
    plt.title('Presicion del Entrenamiento y la Validacion')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Error de entrenamiento')
    plt.plot(epochs_range, val_loss, label='Error de validacion')
    plt.legend(loc='upper right')
    plt.title('Error del entrenamiento y validacion')
    plt.show()

def comprobacion(val_ds,class_names,modelo):

        for i in range(10):
            for images, labels in val_ds.take(1):
                predictions = modelo.predict(images)
                score = tf.nn.softmax(predictions[i])
                print(
                "Esta imagen probablemente pertenece a {}"
                .format(class_names[np.argmax(score)], 100 * np.max(score)))
                plt.imshow(images[i].numpy().astype("uint8"))
                plt.show()
                break
def matriz_confusion(train_ds,class_names,modelo):
    test_labels = []
    test_images = []

    for img, labels in train_ds.take(1):
        test_images.append(img)
        test_labels.append(labels)
        images=np.array(img)
        label=np.array(labels)

    y_pred = np.argmax(modelo.predict(test_images), axis=1).flatten()
    y_true = np.asarray(test_labels).flatten()
    test_acc = sum(y_pred == y_true) / len(y_true)
    print(("Test accuracy: {:.2f}%".format(test_acc * 100)))
    consfusion_matrix = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 10))
    sns.heatmap(consfusion_matrix.numpy(), 
    xticklabels=class_names,
    yticklabels=class_names, 
    annot=True, fmt="d")
    plt.title('Matriz de confusion')
    plt.xlabel('Prediccion')
    plt.ylabel('Real')
    plt.show()

    return images,label

def validacion_cruzada(test_images,test_labels):
     
     keras_model = KerasClassifier(build_fn=funcion_activacion, epochs=100, batch_size=250)
     scores = cross_val_score(keras_model, test_images, test_labels, cv=7)
     mean_score = np.mean(scores)
     plt.bar(range(len(scores)), scores)
     plt.axhline(y=mean_score, color='r', linestyle='-')
     plt.xlabel('Fold')
     plt.ylabel('Score')
     plt.title('Cross-validation scores')
     plt.show()

     
    
if __name__ == '__main__':
    val_ds,train_ds,class_name=cargar_imagenes()
    rotacion_imagenes(train_ds)
    f_activacion=funcion_activacion()
    keras_model = KerasClassifier(build_fn=funcion_activacion, epochs=100, batch_size=250)
    epochs_range ,acc,val_acc,loss,val_loss=entrenamiento(f_activacion)
    graficas(epochs_range ,acc,val_acc,loss,val_loss)
    comprobacion(val_ds,class_name,f_activacion)
    test_images,test_labels=matriz_confusion(train_ds,class_name,f_activacion)
    # validacion_cruzada(test_images,test_labels)
   