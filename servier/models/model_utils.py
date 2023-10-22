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


def create_deep_learning_model(inputs, fully_connected_sizes, scaler, num_classes=None):
    """
    Function that creates that deep learning model that will be used for the classification task
    :param inputs: (Tensor) input tensor
    :param fully_connected_sizes: (Tensor) List containing sizes of the fully connected layers
    :param scaler: (Tensor) scaling Tensor of softmax input
    :param num_classes: (int) number of classes that we are interested in.
    :return: classification (Tensor) , softmax (Tensor)
    """

    model_outputs = inputs

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
