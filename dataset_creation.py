import os
import pandas as pd
import shutil
import numpy as np

dataset_directory = "./datasets/UTKFace"

# impostazione della cartella di esecuzione corretta
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

df = pd.DataFrame()

def extract_info_from_name(directory):
    global df

    files = []
    ages = []
    genders = []
    ethnicities = []

    for file in os.listdir(directory):
        file_name = file.strip(".jpg.chip.jpg")
        file_info = file_name.split("_")

        files.append(file)
        ages.append(int(file_info[0]))
        #genders.append(int(file_info[1]))
        #ethnicities.append(int(file_info[2]))
    
    ethnicity_mapping = {
        0: "White",
        1: "Black",
        2: "Asian",
        3: "Indian",
        4: "Others"
    }

    genders_mapping = {
        0: "Male",
        1: "Female"
    }

    df["file"] = files
    df["age"] = ages
    #df["gender"] = [genders_mapping[value] for value in genders]
    #df["ethnicity"] = [ethnicity_mapping[value] for value in ethnicities]

    df = df[df["age"] < 100]

    fasce = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 80, 100]
    labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', 
          '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-79', '80+']

    # creazione nuova colonna
    df['age_group'] = pd.cut(df['age'], bins=fasce, labels=labels, right=False)

    return df

def save_dataset(directory, df: pd.DataFrame):

    for age_group in df.age_group.unique():
        os.mkdir(f"{directory}/{age_group}")

        subset = df[df["age_group"] == age_group]

        for file in subset["file"]:
            shutil.copy(f"{dataset_directory}/{file}", f"./datasets/ds_standard_2/{age_group}/{file}")

def count_dataset(directory):

    [print(cartella, len(os.listdir(directory+"/"+cartella))) for cartella in os.listdir(directory)]

# Function to create directories if they don't exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Function to split the dataset
def split_dataset(dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Check the ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"

    # Get all class names
    class_names = os.listdir(dataset_dir)

    # Create output directories
    for split in ['train', 'val', 'test']:
        for class_name in class_names:
            create_dir(os.path.join(output_dir, split, class_name))

    # Split each class
    for class_name in class_names:
        class_dir = os.path.join(dataset_dir, class_name)
        images = os.listdir(class_dir)
        np.random.shuffle(images)

        # Split images
        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)
        
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]

        # Copy images to corresponding directories
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, 'train', class_name, img))
        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, 'val', class_name, img))
        for img in test_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(output_dir, 'test', class_name, img))


if __name__ == "__main__":
    df = extract_info_from_name(dataset_directory)

    save_dataset("./datasets/ds_standard_2", df)

    count_dataset("./datasets/ds_standard_2")

    # Usage example
    dataset_dir = './datasets/ds_standard_2'
    output_dir = './datasets/ds_standard_2_split'
    split_dataset(dataset_dir, output_dir)