"""
File that contains code for the model creation management
"""
import os
import sys
import time
import numpy as np
import tf_slim as slim
import tensorflow as tf

# Disable eager mode
tf.compat.v1.disable_eager_execution()
from utils import print_yellow, print_green, print_red


def weight_variables(shape, identifier, trainable=True):
    """
    :param shape: shape of the weight matrix [num_rows,num_cols,depth]
    :param identifier: name of the weight matrix
    :param trainable: freeze variable or modify it through gradient descent, default trainable=True
    :return: tensorflow variable
    """
    initial = tf.random.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
    return tf.Variable(initial, trainable=trainable, name=identifier)


def bias_variables(shape, identifier, trainable=True):
    """
    :param shape: shape of the bias column [num_rows,num_cols,depth]
    :param identifier: name of the bias variable
    :param trainable: freeze variable or modify it through gradient descent, default trainable=True
    :return: tensorflow variable
    """
    initial = tf.constant(1.0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial, trainable=trainable, name=identifier)


def apply_aggregating_squeeze_and_excite(inputs, number_of_matrices):
    """
    This creates and runs the aggregating Squeeze and excite layers
    :param inputs: (N,number_hidden_units) input tensor to apply the operations to
    :param number_of_matrices: Number of matrices to use to compute the weights
    :return: (H.W,) tensor
    :rtype: (Tensor)
    """

    size = inputs.get_shape()[-1]

    # create matrices
    weight_matrices_and_nonlin = create_weight_matrices_and_associated_nonlin(number_of_matrices=number_of_matrices,
                                                                              size=size)
    # apply matrices
    tensor_of_weights = apply_weight_matrices_to_make_tensor_of_weights(inputs, weight_matrices_and_nonlin)

    # use tensor as weights for averaging
    return tf.expand_dims(tf.reduce_mean(tf.multiply(inputs, tensor_of_weights), axis=0), axis=0)


def apply_weight_matrices_to_make_tensor_of_weights(reshaped_input, weight_matrices_and_nonlin):
    """Compute the matrix multiplications and applies the non-linearities in order to produce a learned tensor of weights

    :param reshaped_input: Reshaped and averaged input tensor, of shape (N, H.W)
    :param weight_matrices_and_nonlin: List of matrices and non-linearities to use
    :return: Tensor of weights, of shape (N, 1)
    """
    tensor_of_weights = reshaped_input
    for (weight_matrix, non_linearity) in weight_matrices_and_nonlin:
        tensor_of_weights = non_linearity(tf.matmul(tensor_of_weights, weight_matrix))

    return tensor_of_weights


def create_weight_matrices_and_associated_nonlin(number_of_matrices, size):
    """Creates a given number of learnable matrices and associate a non-linearity function at each
    :param number_of_matrices: Number of matrices to create
    :param size: size of the first dimension of the first matrix
    :return: List of learnable matrices and their non-linearity function
    """
    weight_matrices_and_nonlin = []
    matrices_sizes = get_matrices_sizes(number_of_matrices, size)

    current_size = size

    for i, new_size in enumerate(reversed(matrices_sizes)):
        identifier = f"Aggregation-Squeeze-Matrix-{i + 1}-of-shape-{current_size}-{new_size}"
        matrix = weight_variables(shape=(current_size, new_size), identifier=identifier, trainable=True)

        # As in normal Squeeze and excite, last non lin is a sigmoid
        non_linearity = tf.nn.relu
        if i == number_of_matrices - 1:
            non_linearity = tf.sigmoid

        weight_matrices_and_nonlin.append((matrix, non_linearity))
        current_size = new_size

    return weight_matrices_and_nonlin


def get_matrices_sizes(number_of_matrices, max_size):
    """Gets a linearly distributed list of matrix sizes

    :param number_of_matrices: Number of matrix wanted
    :param max_size: Maximum size of the matrix
    :return: List of matrices' upper sizes
    """
    return np.linspace(1, max_size, number_of_matrices, endpoint=False, dtype=int)


def apply_blstm_layers(inputs, number_of_hidden_units, number_of_layers, number_of_matrices):
    """This creates and runs BLSTM layers over the given inputs.

    :param inputs: (Tensor) Input Tensor to apply the BLSTM layers to.
    :param number_of_hidden_units: (int) Number of hidden units of each internal layer of the BLSTM
    :param number_of_layers: (int) Number of BLSTM layers
    :param number_of_matrices: (int) Number of matrices to apply squeeze and excite layers
    :return: (Tensor) The input tensors to which the BLSTM was applied.
    """
    # Import that allows usage of stackable bidirectional blstm layers
    from models.stack_bidirection_layers_functions import stack_bidirectional_dynamic_rnn

    # First we are going to create the forward and backward cells for each vertical layer
    forward_pass_cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=number_of_hidden_units, use_peepholes=True)
                          for _ in range(number_of_layers)]
    backward_pass_cells = [tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=number_of_hidden_units, use_peepholes=True)
                           for _ in range(number_of_layers)]

    # Once the forward and backward cells have been created we stack the LSTM layers
    outputs, _, _ = stack_bidirectional_dynamic_rnn(
        cells_fw=forward_pass_cells,
        cells_bw=backward_pass_cells,
        inputs=inputs,
        dtype=tf.float32)

    if number_of_matrices:
        outputs = apply_aggregating_squeeze_and_excite(inputs=outputs[0],number_of_matrices=number_of_matrices)
    else:
        # Used To Resize To A Single Array Input
        outputs = tf.reshape(outputs[-1][-1], [-1, 2 * number_of_hidden_units])

    return outputs


def create_deep_learning_model(inputs, use_fingerprint, aggregation_type, aggregation_parameters, fully_connected_sizes,
                               scaler, num_classes=None):
    """
    Function that creates that deep learning model that will be used for the classification task
    :param inputs: (Tensor) input tensor
    :param use_fingerprint: (bool) element that indicates that we are using a naive string
    :param aggregation_type: (str) string specify the aggregation type that is being used
    :param aggregation_parameters: (dict) dictionary containing aggregation types that we are using
    :param fully_connected_sizes: (Tensor) List containing sizes of the fully connected layers
    :param scaler: (Tensor) scaling Tensor of softmax input
    :param num_classes: (int) number of classes that we are interested in.
    :return: classification (Tensor) , softmax (Tensor)
    """

    model_outputs = inputs

    # If we are not using fingerprints, we are using the naive approach
    if not use_fingerprint:
        with tf.name_scope("Input-Aggregation-Layers"):
            if aggregation_type == "blstm":
                model_outputs = tf.expand_dims(input=model_outputs, axis=0)
                model_outputs = apply_blstm_layers(inputs=model_outputs, **aggregation_parameters)

    for matrix_index, matrix_neural_size in enumerate(fully_connected_sizes):
        with tf.name_scope(f"Fully-Connected-Layer-{matrix_index + 1}"):
            fully_connected_weights = weight_variables([model_outputs.get_shape()[-1], matrix_neural_size],
                                                       "Weights", trainable=True)
            fully_connected_bias = bias_variables([matrix_neural_size], "Biases", trainable=True)

            model_outputs = tf.nn.bias_add(tf.matmul(model_outputs, fully_connected_weights), fully_connected_bias)
            model_outputs = tf.nn.relu(model_outputs)

    with tf.name_scope("Classification-Layer"):
        # definition of classification layer
        classifier = weight_variables([fully_connected_sizes[-1], num_classes], "Weights", trainable=True)
        classifier_bias = bias_variables([num_classes], "Biases", trainable=True)

        # output of the neural network before softmax
        classification = tf.nn.bias_add(tf.matmul(model_outputs, classifier), classifier_bias,
                                        name="Classification-output")

    with tf.name_scope("Outputs"):
        # Scaling validation output to keep "dropout scaling coherence during training"
        if scaler:
            softmax = tf.nn.softmax(tf.multiply(classification, scaler), name="Softmax")
        else:
            softmax = tf.nn.softmax(classification, name="Softmax")

    return classification, softmax


def postprocess(classification, label, learning_rate):
    """
    Function that gets minimisation tensor, cross entropy loss as well as accuracy on a forward pas of the network
    :param classification: (Tensor) output from one pass of the deep learning model
    :param label: (Tensor) Ground truth labels of size [batch_size,num_classes]
    :param learning_rate: (Tensor) Learning rate for gradient descent
    :return: Tuple(backward_propagation,cross_entropy,accuracy)
    one backward pass of gradient descent optimization based on forward pass,loss as well as accuracy
    """

    with tf.name_scope("Gradient-Computation"):
        # Loss Definition
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=classification))

        training_step = tf.compat.v1.train.AdamOptimizer(learning_rate, 0.9).minimize(cross_entropy_loss)

    with tf.name_scope("Evaluation-Computation"):
        correct_prediction = tf.equal(tf.argmax(classification, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return training_step, cross_entropy_loss, accuracy


def setup_tensorboard(loss, accuracy, session, tensorboard_path):
    """
    Function that sets up tensorboard viewing
    :param loss: (Tensor) loss tensor
    :param accuracy: (Tensor) accuracy tensor
    :param session: (Tensor) session tensor
    :param tensorboard_path: (str) path to where to store tensors
    :return:
    """
    loss_view = tf.compat.v1.summary.scalar("Loss", loss)
    accuracy_view = tf.compat.v1.summary.scalar("Accuracy", accuracy)

    merged_view = tf.compat.v1.summary.merge_all()

    train_file_writer = tf.compat.v1.summary.FileWriter(tensorboard_path + "/TRAIN", session.graph)
    validation_file_writer = tf.compat.v1.summary.FileWriter(tensorboard_path + "/VALID")

    return merged_view, train_file_writer, validation_file_writer


def restore_last_model(path, session, saver, index_iteration=None, last=True):
    """
    Function that restores the last weights that were used during training for a given iteration
    or a desired index iteration
    :param path: (str) Path where weights are stored
    :param index_iteration: (int) Option to get a model saved at a given iteration
    :param last: (bool) Option to get the last saved model
    :return: (Tensor) returns the index of the last trained model
            If there is no model, we return 0
    """

    ckpt = tf.train.get_checkpoint_state(path, latest_filename="full-checkpoint")
    iteration_last_model = index_iteration if index_iteration else 0

    if last:
        try:
            last_model_path = ckpt.model_checkpoint_path
        except AttributeError:
            iteration_last_model = 0
            last_model_path = None
        else:
            iteration_last_model = int((last_model_path.split("_")[-1]).split(".")[0])

        if os.path.exists(str(last_model_path) + ".meta"):
            print("We Found The Model : ", last_model_path)
            print_yellow("Model weights are being restored.....")
            saver.restore(session, last_model_path)
            print_green("Model weights have been restored")
        else:
            print_red("No Initiation Model Weights Will Be Used...")
            print_green("We generate A New Model That Will Be Trained From Scratch\n")

    return iteration_last_model


def print_model_size():
    """
    Function that prints out the model estimated size
    """
    model_variables = tf.compat.v1.trainable_variables()
    total_size, total_bytes = slim.model_analyzer.analyze_vars(model_variables, print_info=False)
    print_yellow("----------------------------------")
    print_yellow("Estimated Model Size :: %.2f Mo" % (int(total_bytes) / int(1e6)))
    print_yellow("----------------------------------\n")


def console_log_update_tracker(iterations, tracker_dictionary, info_dump):
    """
    :param iterations: (int) Global step during training for a given model
    :param tracker_dictionary: (dict) dictionary containing tracker information
    :param info_dump: (int) number of iteration that were ran from previous information dump
    """
    print("-------------------------------------------------------------")
    print(f"We called the model {iterations} times")
    print(
        f"Moving Average of Training Loss is       : {np.round((tracker_dictionary['training_moving_loss'] / info_dump), 2)}")
    print(
        f"Moving Average of Validation Loss is     : {np.round((tracker_dictionary['validation_moving_loss'] / info_dump), 2)}")
    print(
        f"Moving Average of Training Accuracy is   : {np.round(100 * (tracker_dictionary['training_moving_accuracy'] / info_dump), 2)}%")
    print(
        f"Moving Average of Validation Accuracy is : {np.round(100 * (tracker_dictionary['validation_moving_accuracy'] / info_dump), 2)}%")
    print("These %d Iterations took %d Seconds" % (info_dump, (time.time() - tracker_dictionary['start'])))
    print("-------------------------------------------------------------")

    tracker_dictionary["start"] = time.time()
    tracker_dictionary["training_moving_accuracy"] = 0.0
    tracker_dictionary["validation_moving_accuracy"] = 0.0
    tracker_dictionary["training_moving_loss"] = 0.0
    tracker_dictionary["validation_moving_loss"] = 0.0

    return tracker_dictionary


def manage_error_during_training(iteration, message, saver, session, weight_path):
    """
    Function that saves model when there has been an error encountered
    :param iteration: (int) index iteration of the given model
    :param message: (str) message that we would like to print once the error has been encountered
    :param saver: (Tensor) saver
    :param session: (Tensor) session
    :param weight_path: (str) folder of path where we would like to store the model weights
    :return:
    """
    print_red(message)
    print(f"Saving the current model at iteration {iteration}")
    print_yellow("Model is being saved.....")
    saver.save(session, weight_path + f"/Iteration_{iteration}")
    dump_in_checkpoint(weight_path, iteration)
    print_green("model has been saved successfully")
    session.close()
    sys.exit()


def dump_in_checkpoint(path, model_iteration):
    """
    Function that dumps model iteration in checkpoint file for traceability
    :param path: path where to store the checkpoint file
    :param model_iteration: Index of the model iteration that will be added to the checkpoint file
    :return:
    """
    checkpoint_file = os.path.join(path, "full-checkpoint")
    try:
        with open(checkpoint_file, "r") as f:
            d = f.readlines()
    except FileNotFoundError:
        d = []
    with open(os.path.join(path, "full-checkpoint"), "w") as f:
        if len(d) == 0:
            d.append(f'model_checkpoint_path: "Iteration_{model_iteration}"\n')
            d.append(f'all_model_checkpoint_paths: "Iteration_{model_iteration}"\n')
        else:
            d[0] = f'model_checkpoint_path: "Iteration_{model_iteration}"\n'
            d.append(f'all_model_checkpoint_paths: "Iteration_{model_iteration}"\n')
        for line in d:
            f.write(line)
