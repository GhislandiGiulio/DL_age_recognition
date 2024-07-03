import numpy as np
import pandas as pd

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
plots_dir = "plots"

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


# inizializzazione dataframe con gli array dei dati
df = pd.DataFrame({
    "image_path": path_arr,
    "age": age_arr
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

# creazione grafici distribuzione dell'età 
plt.figure(figsize=(10, 8))
ax = sns.displot(x=df_train["age"])
ax.set_titles("Train - Age")
plt.savefig(f"{plots_dir}/training_age_visualization.jpg")

## validation

# creazione grafici distribuzione dell'età 
plt.figure(figsize=(10, 8))
ax = sns.displot(x=df_valid["age"])
ax.set_titles("Valid - Age")
plt.savefig(f"{plots_dir}/validation_age_visualization.jpg")

## Test

# creazione grafici distribuzione dell'età 
plt.figure(figsize=(10, 8))
ax = sns.displot(x=df_test["age"])
ax.set_titles("Test - Age")
plt.savefig(f"{plots_dir}/test_age_visualization.jpg")

# visualizzazione di un sample del df
def visualize_df(df: np.ndarray):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i, ax in enumerate(axes.ravel()):
        if i < len(df):
            a = np.random.randint(1, len(df), 1)[0]
            img_path = df.loc[a][['image_path']].values[0]
            img_age = df.loc[a][['age']].values[0]
            
            image = Image.open(img_path).convert('RGB')
            
            ax.imshow(image)
            ax.set_title(f"Age: {img_age}")
            ax.axis('off')
            
        else:
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/test_df_visualization.jpg")

visualize_df(df_test)


from tensorflow.keras.preprocessing.image import load_img

# inizializzazione array del training
train_img_arr = []
train_age_arr = []

for idx, row in tqdm(df_train.iterrows()):
    img = load_img(row['image_path'], color_mode="grayscale")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img = np.array(img, dtype=float)

    # riconversione grayscale a     RGB ripetendo il channel
    img = np.repeat(img, CHANNELS, axis=-1)

    img /= 255.0
    train_img_arr.append(img)
    train_age_arr.append(row['age'])
    
train_img_arr = np.array(train_img_arr).reshape(len(train_img_arr), IMG_SIZE, IMG_SIZE, CHANNELS)
train_age_arr = np.array(train_age_arr)


# inizializzazione array del validation
valid_img_arr = []
valid_age_arr = []

for idx, row in tqdm(df_valid.iterrows()):
    img = load_img(row['image_path'], color_mode="grayscale")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    img = np.array(img, dtype=float)

    # riconversione grayscale a RGB ripetendo il channel
    img = np.repeat(img, CHANNELS, axis=-1)

    img /= 255.0
    valid_img_arr.append(img)
    valid_age_arr.append(row['age'])
    
valid_img_arr = np.array(valid_img_arr).reshape(len(valid_img_arr), IMG_SIZE, IMG_SIZE, CHANNELS)
valid_age_arr = np.array(valid_age_arr)


# inizializzazione array del test
test_img_arr = []
test_age_arr = []

for idx, row in tqdm(df_test.iterrows()):
    img = load_img(row['image_path'], color_mode="grayscale")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    # riconversione grayscale a RGB ripetendo il channel
    img = np.repeat(img, CHANNELS, axis=-1)

    img = np.array(img, dtype=float)
    img /= 255.0
    test_img_arr.append(img)
    test_age_arr.append(row['age'])
    
test_img_arr = np.array(test_img_arr).reshape(len(test_img_arr), IMG_SIZE, IMG_SIZE, CHANNELS)
test_age_arr = np.array(test_age_arr)


### 
### Model
###

from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, GlobalAveragePooling2D

model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),

    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),

    # layer di output
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='relu')
])

from keras.src.optimizers import Adam
from tensorflow.python.keras.callbacks import EarlyStopping
from keras.src.callbacks import ReduceLROnPlateau

"""model.compile(optimizer=Adam(learning_rate=1e-3), 
              loss=['mae'], 
              metrics=['mae'])
# lr scheduler per adattare il learning rate in base alle epoche
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=1)
model_es = EarlyStopping(monitor='val_loss', mode='min', patience=3, restore_best_weights=True)

# stampa del modello
model.summary()

# allenamento del modello
history = model.fit(x=train_img_arr, 
                    y=[train_age_arr], 
                    batch_size=BATCH_SIZE, 
                    epochs=EPOCHS, 
                    validation_data=(valid_img_arr, [valid_age_arr]),
                    callbacks=[lr_scheduler, model_es]
                    )

# salvataggio del modello
model.save(f"age_prediction_model2.keras")

###
### Test & Risultati
###

history_df = pd.DataFrame(history.history)
history_df.head()

# plot del MAE per l'età
plt.figure(figsize=(10, 8))

plt.title("Age MAE")

plt.plot(history_df["mae"])
plt.plot(history_df["val_mae"])

plt.legend(["train", "valid"])

plt.savefig(f"{plots_dir}/age_MAE.jpg")"""


## TEST
from tensorflow.keras.models import load_model

model = load_model("age_prediction_model2.keras")

# fase di testing
pred_age = model.predict(test_img_arr, verbose=0)
pred_age = np.round(pred_age).astype(int)
pred_age = pred_age.reshape(-1)

errors = np.abs(pred_age - df_test["age"].values)

mae = np.mean(errors)

print(f"MAE: {mae}")

for i in range(20):
    print(f"Predicted: {pred_age[i]}, Actual: {df_test['age'].values[i]}")

# aggiunto al df di test la colonna predetta
df_test["pred_age"] = pred_age

# funzione per la visualizzazione dei risultati (predizione e true value) in un plot
def visualize_results(df: pd.DataFrame):
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for i, ax in enumerate(axes.ravel()):
        if i < len(df):
            a = np.random.randint(1, len(df), 1)[0]

            img_path = df.loc[a][['image_path']].values[0]
            img_age = df.loc[a][['age']].values[0]
            img_pred_age = df.loc[a][['pred_age']].values[0]
            
            image = Image.open(img_path).convert('RGB')
            
            ax.imshow(image)
            ax.set_title(f"Pred Age: {img_pred_age}(True:{img_age})")
            ax.axis('off')
            
        else:
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/results_plot.jpg")

visualize_results(df_test)