# This is the file that will be used in order to create the training,validation and test database
# for single property prediction
import os
import pandas as pd
from utils import get_train_validation_test_dataframes

# Fetch data folder
app_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
source_folder = os.path.join(app_folder, "instructions")
destination_folder = os.path.join(app_folder, "data", "single")

csv_file = os.path.join(source_folder, "dataset_single.csv")

# Read Data From Single File
total_dataframe = pd.read_csv(csv_file)

(training_dataframe, balanced_training_dataframe, validation_data_frame,
 test_data_frame) = get_train_validation_test_dataframes(total_dataframe)

print("Training Data Class Distribution :\n", training_dataframe["P1"].value_counts())
print("Balanced Training Data Class Distribution :\n", balanced_training_dataframe["P1"].value_counts())
print("Validation Data Class Distribution :\n", validation_data_frame["P1"].value_counts())
print("Test Data Class Distribution :\n", test_data_frame["P1"].value_counts())

training_dataframe.to_csv(os.path.join(destination_folder, "train.csv"), index=False)
balanced_training_dataframe.to_csv(os.path.join(destination_folder, "balanced-train.csv"), index=False)
validation_data_frame.to_csv(os.path.join(destination_folder, "valid.csv"), index=False)
test_data_frame.to_csv(os.path.join(destination_folder, "test.csv"), index=False)
