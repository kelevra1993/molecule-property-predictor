"""
File that contains code for the data management
"""
import os
import numpy as np
import openpyxl as opxl
import tensorflow as tf

# Disable eager mode
tf.compat.v1.disable_eager_execution()
from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops


def get_naive_encoder():
    """
    Function to get naive encoder for smile string characters
    :return: naive_encoding_indices (dict) dictionary containing naive encodings
    """
    naive_encoding_indices = {"2": 0, "[": 1, "c": 2, "S": 3, "n": 4, "4": 5, "F": 6, "/": 7, "+": 8, "o": 9, "#": 10,
                              "]": 11, "H": 12, "N": 13, "C": 14, "s": 15, "(": 16, "l": 17, "=": 18, ")": 19, "-": 20,
                              "5": 21, "3": 22, "B": 23, "\\": 24, "r": 25, "6": 26, "O": 27, "1": 28, "unknown": 29}
    return naive_encoding_indices


def naive_encoder(character):
    """
    Function that encodes a character from a smile string into a vector.
    :param character: (str) this a smile string character
    :return:
    """
    naive_encoding_indices = get_naive_encoder()

    encoding_dimensions = len(naive_encoding_indices)

    if character in naive_encoding_indices:
        encoded_character = create_one_hot_vector(naive_encoding_indices[character], num_classes=encoding_dimensions)
    else:
        encoded_character = create_one_hot_vector(naive_encoding_indices["unknown"], num_classes=encoding_dimensions)

    return encoded_character


def create_dataset(csv_file, column_defaults, field_delimiter):
    """
    Function that is used to create to get the data iteratively from the user's csv file
    :param csv_file: (str) path to csv file
    :param column_defaults: List[Tensors] types of inputs that we get from the csv file
    :param field_delimiter: (str) desired csv field delimiter
    :return:
    """

    dataset = tf.data.experimental.CsvDataset(
        filenames=[csv_file], record_defaults=column_defaults, header=True, field_delim=field_delimiter)

    # repeat the dataset
    dataset = dataset.repeat()

    # Create an iterator
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)

    # Get the next value from the iterator
    value = iterator.get_next()

    # Will be called automatically, but won't need to if we use dataset.repeat
    initializer = iterator.initializer

    return initializer, value


def create_input_producer(csv_file, column_defaults, name_scope, field_delimiter):
    """
    Function that creates the data input producer that can be used for training, validation or testing
    :param csv_file: (str) path to csv file.
    :param column_defaults: List[Tensors] types of inputs that we get from the csv file
    :param name_scope: (str) Name scope that we would like to use.
    :param field_delimiter: (str) delimiter that should be used for data.
    :return: initializer (Tensor), data_tensor (Tensor
    """

    with tf.name_scope("Input-Producer"):
        with tf.name_scope(name_scope):
            initializer, data_tensor = create_dataset(csv_file=csv_file,
                                                      column_defaults=column_defaults,
                                                      field_delimiter=field_delimiter)
    return initializer, data_tensor


def get_model_placeholders(sequence_input_shape, sequence_output_shape):
    """
    Function that gets model placeholders
    :param sequence_input_shape: List[int] list containing the input shape.
    :param sequence_output_shape: List[int] list containing the output shape.
    :return:
    """
    ######################################################
    # We get the placeholders for data that will be used #
    ######################################################
    with tf.name_scope("Placeholders"):
        # todo has to be coded so that we can use batch normalisation !!!!
        with tf.name_scope("Batch-Normalization"):
            normalization_phase = tf.compat.v1.placeholder_with_default(input=True, shape=None)

        with tf.name_scope("Sequence-Input"):
            sequence_data_placeholder = tf.compat.v1.placeholder(tf.float32, shape=sequence_input_shape)

        with tf.name_scope("Sequence-Label"):
            sequence_label_placeholder = tf.compat.v1.placeholder(tf.float32, shape=sequence_output_shape)

    return sequence_data_placeholder, sequence_label_placeholder, normalization_phase


def count_records(session, column_defaults, csv_file, field_delimiter):
    """
    Function that is used to count elements of a given csv file that will later be used.
    :param session: (Tensor) Tensorflow session
    :param column_defaults: List[Tensors] types of inputs that we get from the csv file
    :param csv_file: (str) path to csv file
    :return:
    """

    example_dataset = tf.data.experimental.CsvDataset(
        filenames=[csv_file], record_defaults=column_defaults, header=True, field_delim=field_delimiter)

    example_iterator = tf.compat.v1.data.make_initializable_iterator(example_dataset)

    example_element = example_iterator.get_next()
    example_initializer = example_iterator.initializer

    session.run(example_initializer)

    record_count = 0

    while True:
        try:
            _ = session.run(example_element)
            record_count += 1
        except tf.errors.OutOfRangeError:
            break

    return record_count


def fingerprint_features(smile_string, radius, size, use_chirality, use_bond_types, use_features):
    """
    Function that makes a fingerprint from a smile string
    :param smile_string: (str) smile string of a molecule
    :param radius: (int) considered radius
    :param size: (int) output size of the fingerprint
    :param use_chirality: (bool) choice to use chirality
    :param use_bond_types: (bool) choice to use bond types
    :param use_features: (bool) choice to use features
    :return:
    """
    molecule = MolFromSmiles(smile_string)

    new_order = rdmolfiles.CanonicalRankAtoms(molecule)

    molecule = rdmolops.RenumberAtoms(molecule, new_order)

    # Extract MorganFingerprintAsBitVect
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, radius,
                                                          nBits=size,
                                                          useChirality=use_chirality,
                                                          useBondTypes=use_bond_types,
                                                          useFeatures=use_features)


def create_one_hot_vector(index, num_classes):
    """
    Function that creates a one hot encoding vector
    :param index: (int) index at which we would like to add the 1
    :param num_classes: (int) dimension of our vector
    :return:
    """
    label = np.zeros(num_classes)
    np.put(label, index, 1)

    return label


def prepare_data(data, num_classes,use_fingerprint, fingerprint_type, **kwargs):
    """
    Function that gets data and prepares it in order to be inserted in the neural network
    :param data: (tuple) One line from the csv dataset file
    :param num_classes: (int) number of classes
    :param use_fingerprint: (int) choice to use fingerprint or naive approach
    :param fingerprint_type: (str) fingerprint type
    :param kwargs: (dict) dictionary containing fingerprint parameters
    :return:
    """

    label = create_one_hot_vector(index=int(data[0]), num_classes=num_classes)
    id = data[1].decode("utf-8")
    smile_string = data[2].decode("utf-8")

    if fingerprint_type == "morgan" and use_fingerprint:
        smile_string_fingerprint = fingerprint_features(smile_string,
                                                        radius=kwargs.get("radius"),
                                                        size=kwargs.get("size"),
                                                        use_chirality=kwargs.get("use_chirality"),
                                                        use_bond_types=kwargs.get("use_bond_types"),
                                                        use_features=kwargs.get("use_features"))

        processed_smile_string_input = np.array([int(i) for i in smile_string_fingerprint.ToBitString()])
        processed_smile_string_input = np.expand_dims(processed_smile_string_input,0)
    else:
        processed_smile_string_input = np.array([naive_encoder(character) for character in smile_string])

    return label, id, smile_string, processed_smile_string_input


def dump_info(label_dictionary, data_dictionary, counter_dictionary, output_file, template_path, scaler):
    """
    Function that dumps metric information into an excel file for analysis
    :param label_dictionary: (dict)project label dictionary
    :param data_dictionary: (dict) data dictionary from evaluation
    :param counter_dictionary: (dict) counter of images per label
    :param output_file: (str) desired path for results
    :param template_path: (str) path of the template file
    :return: dump all information about a model's evaluation in an excel file
    """

    # Load template workbook
    wb = opxl.load_workbook(template_path, keep_vba=True)

    # First we create worksheets for a given label in a label dictionary
    for i in range(len(label_dictionary) - 1):
        buffer_sheet = wb["Classifieur Binaire"]
        wb.copy_worksheet(buffer_sheet)

    # Rename worksheets
    for en, sh in enumerate(wb):
        sh.title = str(label_dictionary[en])

    # sorting following the predicted class output
    for k in data_dictionary:
        data_dictionary[k].sort(key=lambda tup: tup[2])
        data_dictionary[k].reverse()
        sheet = wb[k]
        for index, info in enumerate(data_dictionary[k]):
            sheet["A" + str(index + 2)].value = info[0]
            sheet["B" + str(index + 2)].value = info[1]
            sheet["C" + str(index + 2)].value = info[2]

        sheet["D" + str(2)].value = str(scaler) if scaler else "1.0"
        sheet["D" + str(51)].value = counter_dictionary[k]

    # Computation of Specificity which is equal to TN/(TN+FP)
    for k in data_dictionary:
        sheet = wb[k]
        N = "+".join(["%s!$D$51" % l for l in label_dictionary.values() if l != k])
        FP = "-".join(["$E$31", "$G$31"])
        # Number of True Negatives of class k
        TN = "(" + N + "-" + "(" + FP + ")" + ")"
        sheet["F" + str(51)].value = "=100*" + TN + "/" + "(" + N + ")"

    wb.save(output_file)
    wb.close()
