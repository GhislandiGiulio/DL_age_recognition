import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

# impostazione della cartella di esecuzione corretta
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

import numpy as np
import pandas as pd

import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")

from PIL import Image

# disattivazione dei warning 
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


###
### Preprocessing
###

# caricamento dataset

root_dir = "datasets/crop_part1"

# variabili di DL
BATCH_SIZE = 32
IMG_SIZE = 128
CHANNELS = 1
EPOCHS = 30


# inizializzazione array nomi file, età e generi
path_arr = []
age_arr = []
gender_arr = []

# estrazione dati dai nomi dei file
for file in tqdm(os.listdir(root_dir)):
    labels = file.split("_")
    
    filepath = os.path.join(root_dir, file)
    age = int(labels[0])
    gender = int(labels[1])
    
    path_arr.append(filepath)
    age_arr.append(age)
    gender_arr.append(gender)


# inizializzazione dataframe con gli array dei dati
df = pd.DataFrame({
    "image_path": path_arr,
    "age": age_arr,
    "gender": gender_arr
})

# shuffle del dataframe
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# definizione delle percentuale di split
train_frac = 0.6
valid_frac = 0.2

# calcolo indici per lo splitting
train_idx = int(train_frac * len(df))
valid_idx = int((train_frac + valid_frac) * len(df))

# divisione dataframe in train, val e test
df_train, df_valid, df_test = np.split(df_shuffled, [train_idx, valid_idx])

df_train = df_train.reset_index(drop=True)
df_valid = df_valid.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print("Total Size:", df.shape[0])
print("Train Size:", df_train.shape[0])
print("valid Size:", df_valid.shape[0])
print("Test Size:", df_test.shape[0])

###
### Data Visualization
###

## training

# creazione grafici distribuzione del genere
plt.figure(figsize=(10, 8))
ax = sns.countplot(x="gender", order=df_train["gender"].value_counts().index, data=df_train)
ax.set_title("Train - Gender")
for container in ax.containers:
    ax.bar_label(container)
plt.savefig("training_gender_visualization.jpg")

# creazione grafici distribuzione dell'età 
plt.figure(figsize=(10, 8))
ax = sns.displot(x=df_train["age"])
ax.set_titles("Train - Age")
plt.savefig("training_age_visualization.jpg")

## validation

# creazione grafici distribuzione dell'età
plt.figure(figsize=(10, 8))
ax = sns.countplot(x="gender", order=df_valid["gender"].value_counts().index, data=df_valid)
ax.set_title("Valid - Gender")
for container in ax.containers:
    ax.bar_label(container)
plt.savefig("validation_gender_visualization.jpg")

# creazione grafici distribuzione dell'età 
plt.figure(figsize=(10, 8))
ax = sns.displot(x=df_valid["age"])
ax.set_titles("Valid - Age")
plt.savefig("validation_age_visualization.jpg")

## Test

# creazione grafici distribuzione dell'età
plt.figure(figsize=(10, 8))
ax = sns.countplot(x="gender", order=df_test["gender"].value_counts().index, data=df_test)
ax.set_title("Test - Gender")
for container in ax.containers:
    ax.bar_label(container)
plt.savefig("test_gender_visualization.jpg")

# creazione grafici distribuzione dell'età 
plt.figure(figsize=(10, 8))
ax = sns.displot(x=df_test["age"])
ax.set_titles("Test - Age")
plt.savefig("test_age_visualization.jpg")


# visualizzazione di un sample del df
def visualize_df(df: np.ndarray):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i, ax in enumerate(axes.ravel()):
        if i < len(df):
            a = np.random.randint(1, len(df), 1)[0]
            img_path = df.loc[a][['image_path']].values[0]
            img_age = df.loc[a][['age']].values[0]
            img_gender = df.loc[a][['gender']].values[0]
            
            image = Image.open(img_path).convert('RGB')
            
            ax.imshow(image)
            ax.set_title(f"Age: {img_age}\nGender: {img_gender}")
            ax.axis('off')
            
        else:
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig("test_df_visualization.jpg")

visualize_df(df_test)


from tensorflow.keras.preprocessing.image import load_img

# inizializzazione array del training
train_img_arr = []
train_gender_arr = []
train_age_arr = []

for idx, row in tqdm(df_train.iterrows()):
    img = load_img(row['image_path'], color_mode="grayscale")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img = np.array(img, dtype=float)

    # riconversione grayscale a     RGB ripetendo il channel
    img = np.repeat(img, CHANNELS, axis=-1)

    img /= 255.0
    train_img_arr.append(img)
    train_gender_arr.append(row['gender'])
    train_age_arr.append(row['age'])
    
train_img_arr = np.array(train_img_arr).reshape(len(train_img_arr), IMG_SIZE, IMG_SIZE, CHANNELS)
train_gender_arr = np.array(train_gender_arr)
train_age_arr = np.array(train_age_arr)


# inizializzazione array del validation
valid_img_arr = []
valid_gender_arr = []
valid_age_arr = []

for idx, row in tqdm(df_valid.iterrows()):
    img = load_img(row['image_path'], color_mode="grayscale")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img = np.array(img, dtype=float)

    # riconversione grayscale a RGB ripetendo il channel
    img = np.repeat(img, CHANNELS, axis=-1)

    img /= 255.0
    valid_img_arr.append(img)
    valid_gender_arr.append(row['gender'])
    valid_age_arr.append(row['age'])
    
valid_img_arr = np.array(valid_img_arr).reshape(len(valid_img_arr), IMG_SIZE, IMG_SIZE, CHANNELS)
valid_gender_arr = np.array(valid_gender_arr)
valid_age_arr = np.array(valid_age_arr)


# inizializzazione array del test
test_img_arr = []
test_gender_arr = []
test_age_arr = []

for idx, row in tqdm(df_test.iterrows()):
    img = load_img(row['image_path'], color_mode="grayscale")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    # riconversione grayscale a RGB ripetendo il channel
    img = np.repeat(img, CHANNELS, axis=-1)

    img = np.array(img, dtype=float)
    img /= 255.0
    test_img_arr.append(img)
    test_gender_arr.append(row['gender'])
    test_age_arr.append(row['age'])
    
test_img_arr = np.array(test_img_arr).reshape(len(test_img_arr), IMG_SIZE, IMG_SIZE, CHANNELS)
test_gender_arr = np.array(test_gender_arr)
test_age_arr = np.array(test_age_arr)


### 
### Model
###

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, GlobalAveragePooling2D

convolution = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),

    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten()
])

# definizione branch del genere
gender_branch = Sequential([
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

# definizione branch dell'età
age_branch = Sequential([
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='relu')
])

# creazione dell'input
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

# passaggio attraverso la convoluzione
flattened_image = convolution(inputs)

# passaggio attraverso i due branch
output_gender = gender_branch(flattened_image)
output_age = age_branch(flattened_image)

# definizione modello
model = Model(inputs=[inputs], outputs=[output_gender, output_age])

from keras.src.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.src.callbacks import ReduceLROnPlateau

model.compile(optimizer=Adam(learning_rate=1e-3), 
              loss=['binary_crossentropy', 'mae'], 
              metrics=['accuracy', 'mae'])
# lr scheduler per adattare il learning rate in base alle epoche
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
model_es = EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)

# stampa del modello
model.summary()

# allenamento del modello
history = model.fit(x=train_img_arr, 
                    y=[train_gender_arr, train_age_arr], 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, 
                    validation_data=(valid_img_arr, [valid_gender_arr, valid_age_arr]),
                    callbacks=[lr_scheduler, model_es]
                    )

# salvataggio del modello
model.save("age_sex_prediction_model.keras")

###
### Test & Risultati
###

history_df = pd.DataFrame(history.history)
history_df.head()

# plot dell'accuracy del genere
plt.figure(figsize=(10, 8))

plt.title("Gender Accuracy")

plt.plot(history_df["sequential_1_accuracy"])
plt.plot(history_df["val_sequential_1_accuracy"])

plt.legend(["train", "valid"])

plt.savefig("gender_accuracy.jpg")


# plot del MAE per l'età
plt.figure(figsize=(10, 8))

plt.title("Age MAE")

plt.plot(history_df["sequential_2_mae"])
plt.plot(history_df["val_sequential_2_mae"])

plt.legend(["train", "valid"])

plt.savefig("age_MAE.jpg")


## TEST

# fase di testing
predictions = model.predict(test_img_arr, verbose=0)
pred_gender = np.argmax(predictions[0], axis=1)
pred_age = [round(prediction[0]) for prediction in predictions[1]]

print(pred_gender)

df_test["pred_age"] = pred_age
df_test["pred_gender"] = pred_gender

# funzione per la visualizzazione dei risultati (predizione e true value) in un plot
def visualize_results(df: pd.DataFrame):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i, ax in enumerate(axes.ravel()):
        if i < len(df):
            a = np.random.randint(1, len(df), 1)[0]
            gender_dict = {0: 'Male', 1: 'Female'}
            img_path = df.loc[a][['image_path']].values[0]
            img_age = df.loc[a][['age']].values[0]
            img_gender = df.loc[a][['gender']].values[0]
            img_pred_age = df.loc[a][['pred_age']].values[0]
            img_pred_gender = df.loc[a][['pred_gender']].values[0]
            
            image = Image.open(img_path).convert('RGB')
            
            ax.imshow(image)
            ax.set_title(f"Pred Age: {img_pred_age}(True:{img_age})\nPred Gender: {gender_dict[img_pred_gender]}(True:{gender_dict[img_gender]})")
            ax.axis('off')
            
        else:
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig("results_plot.jpg")

visualize_results(df_test)