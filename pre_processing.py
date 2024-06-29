import os
import pandas as pd

dataset_directory = "./datasets/crop_part1"

# impostazione della cartella di esecuzione corretta
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

print(len(os.listdir("datasets/crop_part1")))
print(len(os.listdir("datasets/UTKFace")))

def estrai_info_da_nome(directory):
    for file_name in os.listdir(directory):
        file_name = file_name.strip(".jpg")
        file_info = file_name.split("_")

        file_info[0] = 


if __name__ == "__main__":
    estrai_info_da_nome(dataset_directory)