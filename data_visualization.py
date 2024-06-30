import pandas as pd

data = pd.read_csv("data.csv")

import os

# impostazione della cartella di esecuzione corretta
script_dir = os.path.dirname(__file__)
os.chdir(script_dir)

import matplotlib.pyplot as plt
import seaborn as sns

import seaborn as sns

age_group_order = [
    '0-4', '5-9', '10-14', '15-19', 
    '20-24', '25-29', '30-34', '35-39', 
    '40-44', '45-49', '50-54', '55-59', 
    '60-64', '65-69', '70-74', '75-79', 
    '80-84', '85-89', '90-94', '95-99', 
    '100-104'
]

# Set the style of the plots
sns.set_theme(style="whitegrid")

# Distribution of Ages
sns.catplot(data=data, kind="count", x="age_group", height=6, aspect=2, order=age_group_order)
sns.despine(left=True)
plt.title('Distribution of Ages')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()

# Distribution of Gender
sns.catplot(data=data, kind="count", x="gender", height=4, aspect=1.5)
sns.despine(left=True)
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

# Distribution of Ethnicity
sns.catplot(data=data, kind="count", x="ethnicity", height=6, aspect=2)
sns.despine(left=True)
plt.title('Distribution of Ethnicity')
plt.xlabel('Ethnicity')
plt.ylabel('Count')
plt.show()
