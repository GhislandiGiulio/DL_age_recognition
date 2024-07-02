#!/usr/bin/env python
# coding: utf-8

# In[11]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

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
sns.set_style("whitegrid")
import PIL
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import load_img
#from tensorflow.keras.utils import plot_model

###
### Preprocessing
###

# caricamento dataset

root_dir = "datasets/crop_part1"

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

# creazione grafici distribuzione dell'età 
plt.figure(figsize=(10, 8))
ax = sns.displot(x=df_train["age"])
ax.set_titles("Train - Age")
plt.show()

## validation

# creazione grafici distribuzione dell'età
plt.figure(figsize=(10, 8))
ax = sns.countplot(x="gender", order=df_valid["gender"].value_counts().index, data=df_valid)
ax.set_title("Valid - Gender")
for container in ax.containers:
    ax.bar_label(container)

# creazione grafici distribuzione dell'età 
plt.figure(figsize=(10, 8))
ax = sns.displot(x=df_valid["age"])
ax.set_titles("Valid - Age")
plt.show()

## Test

# creazione grafici distribuzione dell'età
plt.figure(figsize=(10, 8))
ax = sns.countplot(x="gender", order=df_test["gender"].value_counts().index, data=df_test)
ax.set_title("Test - Gender")
for container in ax.containers:
    ax.bar_label(container)

# creazione grafici distribuzione dell'età 
plt.figure(figsize=(10, 8))
ax = sns.displot(x=df_test["age"])
ax.set_titles("Test - Age")
plt.show()


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
    plt.show()

visualize_df(df_train)



# inizializzazione array del training
train_img_arr = []
train_gender_arr = []
train_age_arr = []

for idx, row in tqdm(df_train.iterrows()):
    img = load_img(row['image_path'], color_mode="grayscale")
    img = img.resize((128, 128), Image.LANCZOS)
    img = np.array(img, dtype=float)
    img /= 255.0
    train_img_arr.append(img)
    train_gender_arr.append(row['gender'])
    train_age_arr.append(row['age'])
    
train_img_arr = np.array(train_img_arr).reshape(len(train_img_arr), 128, 128, 1)
train_gender_arr = np.array(train_gender_arr)
train_age_arr = np.array(train_age_arr)


# inizializzazione array del validation
valid_img_arr = []
valid_gender_arr = []
valid_age_arr = []

for idx, row in tqdm(df_valid.iterrows()):
    img = load_img(row['image_path'], color_mode="grayscale")
    img = img.resize((128, 128), Image.LANCZOS)
    img = np.array(img, dtype=float)
    img /= 255.0
    valid_img_arr.append(img)
    valid_gender_arr.append(row['gender'])
    valid_age_arr.append(row['age'])
    
valid_img_arr = np.array(valid_img_arr).reshape(len(valid_img_arr), 128, 128, 1)
valid_gender_arr = np.array(valid_gender_arr)
valid_age_arr = np.array(valid_age_arr)


# inizializzazione array del test
test_img_arr = []
test_gender_arr = []
test_age_arr = []

for idx, row in tqdm(df_test.iterrows()):
    img = load_img(row['image_path'], color_mode="grayscale")
    img = img.resize((128, 128), Image.LANCZOS)
    img = np.array(img, dtype=float)
    img /= 255.0
    test_img_arr.append(img)
    test_gender_arr.append(row['gender'])
    test_age_arr.append(row['age'])
    
test_img_arr = np.array(test_img_arr).reshape(len(test_img_arr), 128, 128, 1)
test_gender_arr = np.array(test_gender_arr)
test_age_arr = np.array(test_age_arr)


### 
### Model
###

from tensorflow.keras.models import Sequential

convolution = Sequential([
    Input(shape=(128, 128, 1)),

    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten()
])

gender_branch = Sequential([
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

# Define the age branch
age_branch = Sequential([
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='relu')
])

# creazione dell'input
inputs = Input(shape=(128, 128, 1))

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
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
model_es = EarlyStopping(monitor='val_loss', mode='min', patience=2, restore_best_weights=True)

# stampa del modello
model.summary()

# allenamento del modello
history = model.fit(x=train_img_arr, 
                    y=[train_gender_arr, train_age_arr], 
                    batch_size=64, 
                    epochs=25, 
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


# In[ ]:


plt.figure(figsize=(10, 8))

plt.title("Gender Accuracy")

plt.plot(history_df["sequential_1_accuracy"])
plt.plot(history_df["val_sequential_1_accuracy"])

plt.legend(["train", "valid"])

plt.show()


# In[ ]:


plt.figure(figsize=(10, 8))

plt.title("Age MAE")

plt.plot(history_df["sequential_2_mae"])
plt.plot(history_df["val_sequential_2_mae"])

plt.legend(["train", "valid"])

plt.show()


# # Test

# In[ ]:


preds = model.predict(test_img_arr, verbose=0)
pred_gender = np.argmax(preds[0], axis=1)
pred_age = [round(pred[0]) for pred in preds[1]]


# In[ ]:


df_test["pred_age"] = pred_age
df_test["pred_gender"] = pred_gender


# In[ ]:


df_test.head()


# In[ ]:


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


# In[ ]:


visualize_results(df_test)


# In[ ]:




