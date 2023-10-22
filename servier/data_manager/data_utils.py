"""
File that contains code for the data management
"""
import tensorflow as tf


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
