import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import os

# impostazione della cartella di esecuzione corretta
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

# definizione della directory contenente il dataset
dataset_directory = "./datasets/crop_part1"

def gender_encode(data):
    # conversione gender a numerico
    data['gender'] = data['gender'].map({'Female': 0, 'Male': 1})

    return data

def limit_samples_per_class(df, feature, max_samples):
    grouped = df.groupby(feature)
    limited_df = grouped.apply(lambda x: x.sample(n=min(len(x), max_samples), random_state=42))
    limited_df = limited_df.reset_index(drop=True)
    return limited_df

def one_hot_encode(data, feature_name):

    # one-hot encoding per l'etnia
    dummy_features = pd.get_dummies(data[feature_name], prefix=feature_name)
    data = pd.concat([data, dummy_features], axis=1)

    return data

def data_train_val_test_subdivision(data):
    X = data.drop(columns=["age"])
    y = pd.DataFrame(data["age_group"])

    # divisione in train e temp (che verrà diviso ulteriormente in val e test)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y,
                                        test_size=0.3, 
                                        random_state=42, 
                                        stratify=data[["ethnicity", "age_group"]]
                                        )

    # divisione in val e test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, 
                                    test_size=0.5, 
                                    random_state=42, 
                                    stratify=X_temp[["age_group"]])
    print(X_val.shape, y_val.shape, X_test.shape, y_test.shape)
    
    X_train.drop(columns=["ethnicity", "age_group"], inplace=True)
    X_val.drop(columns=["ethnicity", "age_group"], inplace=True)
    X_test.drop(columns=["ethnicity", "age_group"], inplace=True)

    return X_train, y_train, X_val, y_val, X_test, y_test


def __preprocess_image(image_path):
    global dataset_directory

    image_height = image_width = 220

    img_array = None

    # Open the image using Pillow
    with Image.open(f"{dataset_directory}/{image_path}") as img:
        # convert to greyscale
        img = img.convert('L')
        # Resize the image
        img = img.resize((image_width, image_height))
        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0

    return img_array

# Example: Load and preprocess data function
def load_and_preprocess_images(image_paths):

    images = np.array([__preprocess_image(path) for path in image_paths])
    return images


def preprocessing():

    # Load the CSV file
    data = pd.read_csv("data.csv")

    data = limit_samples_per_class(data, "age_group", 174)

    data = gender_encode(data)

    data = one_hot_encode(data, "ethnicity")

    X_train, y_train, X_val, y_val, X_test, y_test = data_train_val_test_subdivision(data)

    y_train = one_hot_encode(y_train, "age_group")
    y_train.drop(columns=["age_group"], inplace=True)
    y_train = y_train.astype(np.float32)

    y_val = one_hot_encode(y_val, "age_group")
    y_val.drop(columns=["age_group"], inplace=True)
    y_val = y_val.astype(np.float32)

    y_test = one_hot_encode(y_test, "age_group")
    y_test.drop(columns=["age_group"], inplace=True)
    y_test = y_test.astype(np.float32)


    # preprocessing delle immagini
    train_images = load_and_preprocess_images(X_train["file"])

    val_images = load_and_preprocess_images(X_val["file"])

    test_images = load_and_preprocess_images(X_test["file"])

    # drop delle colonne file
    X_train.drop(columns=["file"], inplace=True)
    X_val.drop(columns=["file"], inplace=True)
    X_test.drop(columns=["file"], inplace=True)

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test, train_images, val_images, test_images