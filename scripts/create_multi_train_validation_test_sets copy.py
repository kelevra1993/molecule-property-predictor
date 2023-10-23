# This is the file that will be used in order to create the training,validation and test databases
# for multi property prediction.
import os
import pandas as pd
from utils import get_multi_train_validation_test_dataframes

# Fetch data folder
app_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
source_folder = os.path.join(app_folder, "instructions")
destination_folder = os.path.join(app_folder, "data", "multi")

csv_file = os.path.join(source_folder, "dataset_multi.csv")

# Read Data From Single File
total_dataframe = pd.read_csv(csv_file)

(training_dataframe, validation_data_frame,
 test_data_frame) = get_multi_train_validation_test_dataframes(total_dataframe)

for i in range(1, 10):
    print(50*'-')
    print("Training Data Class Distribution :\n", training_dataframe[f"P{i}"].value_counts())
    print("Validation Data Class Distribution :\n", validation_data_frame[f"P{i}"].value_counts())
    print("Test Data Class Distribution :\n", test_data_frame[f"P{i}"].value_counts())
    print(50 * '-')

training_dataframe.to_csv(os.path.join(destination_folder, "train.csv"), index=False)
validation_data_frame.to_csv(os.path.join(destination_folder, "valid.csv"), index=False)
test_data_frame.to_csv(os.path.join(destination_folder, "test.csv"), index=False)
