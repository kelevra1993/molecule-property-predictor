import numpy as np
import pandas as pd
from collections import Counter

np.set_printoptions(linewidth=200)


def split_dataframe_by_percentage(dataframe, percent):
    """
    Splits a DataFrame into two parts based on a percentage value.
    :param dataframe: (DataFrame) The input DataFrame to be split.
    :param percent: (float) The percentage of rows to be included in the first part.
    :return: (DataFrame, DataFrame) A tuple containing two DataFrames:
             - The first DataFrame with the specified percentage of rows.
             - The second DataFrame with the remaining rows.
    """
    num_rows_df1 = int(len(dataframe) * (percent / 100))
    df1 = dataframe.iloc[:num_rows_df1]
    df2 = dataframe.iloc[num_rows_df1:]

    return df1, df2


def get_train_validation_test_dataframes(total_dataframe):
    """
    Splits the total DataFrame into training, validation, and test DataFrames and balances the training data.
    :param total_dataframe: (DataFrame) The total input DataFrame to be split into training, validation, and test sets.
    :return: (DataFrame, DataFrame, DataFrame, DataFrame) A tuple containing four DataFrames:
             - The training DataFrame.
             - The balanced training DataFrame with rows corresponding to the minority class duplicated.
             - The validation DataFrame.
             - The test DataFrame.
    """
    training_dataframe, validation_test_data_frame = split_dataframe_by_percentage(total_dataframe, percent=80)

    balanced_training_dataframe = balance_dataframe(training_dataframe, column_name="P1")
    validation_data_frame, test_data_frame = split_dataframe_by_percentage(validation_test_data_frame, percent=50)

    return training_dataframe, balanced_training_dataframe, validation_data_frame, test_data_frame


def get_multi_train_validation_test_dataframes(total_dataframe):
    """
    Splits the total DataFrame into training, validation, and test DataFrames and balances the training data.
    :param total_dataframe: (DataFrame) The total input DataFrame to be split into training, validation, and test sets.
    :return: (DataFrame, DataFrame, DataFrame, DataFrame) A tuple containing four DataFrames:
             - The training DataFrame.
             - The validation DataFrame.
             - The test DataFrame.
    """
    training_dataframe, validation_test_data_frame = split_dataframe_by_percentage(total_dataframe, percent=80)
    validation_data_frame, test_data_frame = split_dataframe_by_percentage(validation_test_data_frame, percent=50)

    return training_dataframe, validation_data_frame, test_data_frame


def balance_dataframe(dataframe, column_name):
    """
    Balances a DataFrame by duplicating rows corresponding to the minority class in a specified column.
    :param dataframe: (DataFrame) The input DataFrame to be balanced.
    :param column_name: (str) The name of the column to balance based on.
    :return: (DataFrame) A new DataFrame where rows corresponding to the minority class are duplicated,
             resulting in a balanced dataset.
    """
    # Find the value counts for the first column
    value_counts = dataframe[column_name].value_counts()

    # Identify the minority class
    minority_class = value_counts.idxmin()

    # Calculate the number of times to duplicate the minority class
    max_count = value_counts.max()
    duplication_factor = max_count // value_counts[minority_class]

    # Create a list of DataFrames with duplicated rows
    duplicated_frames = [dataframe[dataframe[column_name] == minority_class]] * duplication_factor

    # Concatenate the original DataFrame and the duplicated DataFrames
    balanced_df = pd.concat([dataframe] + duplicated_frames)

    balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

    return balanced_df


def extract_unique_characters(dataframe, column_name):
    """
    Extracts all unique characters from a DataFrame column containing strings.

    :param dataframe: (DataFrame) The input DataFrame.
    :param column_name: (str) The name of the column containing strings.
    :return: (list) A sorted list of unique characters found in the specified column.
    """
    # Create an empty set to store unique characters
    unique_chars = set()

    # Iterate through the column and extract characters
    for string in dataframe[column_name]:
        unique_chars.update(set(string))

    # Convert the set to a sorted list
    unique_chars_list = sorted(list(unique_chars))

    return unique_chars_list


def have_same_elements(list1, list2):
    """
    Check if two lists have the same elements, regardless of their order.
    :param list1: (list) The first list.
    :param list2: (list) The second list.
    :return: (bool) True if both lists have the same elements, False otherwise.
    """
    return Counter(list1) == Counter(list2)


def create_one_hot_vector(index, num_classes):
    """
    Function that creates a one hot encoding vector
    :param index: (int) index at which we would like to add the 1
    :param num_classes: (int) dimension of our vector
    :return:
    """
    label = np.zeros(num_classes, dtype=np.float32)
    np.put(label, index, float(1.0))

    return label
