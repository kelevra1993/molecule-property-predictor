'''
This file will be used in order to analyse all of the smile strings and for each element (string) it will allocate an index

For instance :
- C (carbon) would be allocated index 1
- c (carbon in an aromatic ring) would be allocated index 2
- @ ( chirality type ) would be allocated index 3

Note : Keep in mind that there is no interpretation that is one here
"@" and "@@" are not separated in the naive setting, just as there is no linkage between "C" and "c" despite both being carbons
"Cl" despite being chlorine would be interpreted as "C" then "l"
and so on...
'''
import os
import json
import pandas as pd
from utils import extract_unique_characters

# Fetch data folder
app_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
source_folder = os.path.join(app_folder, "instructions")
destination_folder = os.path.join(app_folder, "data", "single")

csv_single_file = os.path.join(source_folder, "dataset_single.csv")
csv_multiple_file = os.path.join(source_folder, "dataset_multi.csv")

# Read Data From Single and Multiple File
total_single_dataframe = pd.read_csv(csv_single_file)
total_multiple_dataframe = pd.read_csv(csv_multiple_file)

# Get unique number of all possible known characters in our dataset
# ( keep in mind that we do not have all atoms as well as some properties of smile strings )
# It will have to be accounted for
unique_characters_from_single = extract_unique_characters(dataframe=total_single_dataframe, column_name="smiles")
unique_characters_from_multiple = extract_unique_characters(dataframe=total_multiple_dataframe, column_name="smiles")

all_possible_characters = list(set(unique_characters_from_single + unique_characters_from_multiple))

# Make the one hot encoder dictionary, accounting for future elements that are unknown
# by putting them in the same one hot dimension, that is added
number_of_dimensions = len(all_possible_characters) + 1

naive_dictionary_encoder = {character: index for index, character in enumerate(all_possible_characters)}
naive_dictionary_encoder["unknown"] = number_of_dimensions - 1

with open("naive_encodings.json", "w") as output_json:
    json.dump(naive_dictionary_encoder, output_json)
