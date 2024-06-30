import os
import pandas as pd

dataset_directory = "./datasets/crop_part1"

# impostazione della cartella di esecuzione corretta
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

print(len(os.listdir("datasets/crop_part1")))
print(len(os.listdir("datasets/UTKFace")))

df = pd.DataFrame()

def extract_info_from_name(directory):
    global df

    files = []
    ages = []
    genders = []
    ethnicities = []

    for file in os.listdir(directory):
        file_name = file.strip(".jpg")
        file_info = file_name.split("_")

        files.append(file)
        ages.append(int(file_info[0]))
        genders.append(int(file_info[1]))
        ethnicities.append(int(file_info[2]))
    
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
    df["gender"] = [genders_mapping[value] for value in genders]
    df["ethnicity"] = [ethnicity_mapping[value] for value in ethnicities]

    df = df[df["age"] < 100]
    df = df[df["ethnicity"] == "White"]

    fasce = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 100]
    labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', 
          '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70+']

    # creazione nuova colonna
    df['age_group'] = pd.cut(df['age'], bins=fasce, labels=labels, right=False)

    [print(f"{age_group}, {len(df[df["age_group"] == age_group])}") for age_group in df["age_group"].unique()]

    df.to_csv("data.csv")

def save_dataset(directory, df: pd.DataFrame):

    for feature in df.columns.values:
        print(feature)



if __name__ == "__main__":
    extract_info_from_name(dataset_directory)