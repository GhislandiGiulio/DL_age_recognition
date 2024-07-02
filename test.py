import matplotlib.pyplot as plt
import numpy as np
from keras.src.callbacks import ReduceLROnPlateau
from keras.src.layers import Conv2D, MaxPooling2D
from keras.src.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import glob
import random
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models
import tensorflow as tf


import os

# impostazione della cartella di esecuzione corretta
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

tf.random.set_seed(0)

train_dir = './Data/ASL_dataset3/Train_Alphabet'

# definizione dei parametri in ingresso
IMAGE_SIZE = 224
num_classes = 16
target_size = (IMAGE_SIZE, IMAGE_SIZE)
target_dims = (IMAGE_SIZE, IMAGE_SIZE, 3)
BATCH_SIZE = 32

from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.layers import RandomRotation
from tensorflow.keras.layers import RandomZoom
from tensorflow.keras.layers import RandomContrast

data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2)
])

def augment(image, label):
    image = data_augmentation(image)
    return image, label


# Creazione dei generatori
# preprocessing dei dati e normalizzazione, data augmentation
train_data_dir = "datasets/ds_standard_2_split/train"
validation_data_dir = "datasets/ds_standard_2_split/val"


train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        train_data_dir,
        label_mode='categorical', 
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        #color_mode='grayscale',
        batch_size=BATCH_SIZE,
        
    )

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        validation_data_dir,
        label_mode='categorical', 
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        #color_mode='grayscale',
        batch_size=BATCH_SIZE
    )

class_names = val_dataset.class_names

train_dataset = train_dataset.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.map(augment, num_parallel_calls=tf.data.experimental.AUTOTUNE)


# visualizzaizone di un sample per classe
"""classi = [folder[len(train_dir) + 1:] for folder in glob.glob(train_data_dir + '/*')]
classi.sort()"""


"""def grafico_classi(base_path):
    colonne = 5
    righe = int(np.ceil(len(classi) / colonne))
    fig = plt.figure(figsize=(16, 20))

    for i in range(len(classi)):
        cls = classi[i]
        img_path = base_path + '/' + cls + '/**'
        path_contents = glob.glob(img_path)

        immagini = random.sample(path_contents, 1)

        sp = plt.subplot(righe, colonne, i + 1)
        plt.imshow(cv2.imread(immagini[0])[:, :, ::-1])  # Convert BGR to RGB
        plt.title(cls)
        sp.axis('off')

    plt.show()"""


#grafico_classi(train_dir)

base_model = tf.keras.applications.ResNet50(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model
base_model.trainable = False

# Add custom layers on top
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Use the appropriate number of classes
])
 #print(model.summary())

model.compile(optimizer=Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
# lr scheduler per adattare il learning rate in base alle epoche
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
model_es = EarlyStopping(monitor='val_loss', mode='min', patience=2, restore_best_weights=True)

# training del modello
history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=[lr_scheduler, model_es]
)

model.save('PROJECT106.keras')

# Plot co i risultati del training
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy del modello')
plt.ylabel('Accuracy')
plt.xlabel('Epoche')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig("accuracy.jpg")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig("loss.jpg")
