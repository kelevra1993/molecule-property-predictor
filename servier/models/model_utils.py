"""
File that contains code for the model creation management
"""
import tensorflow as tf


def weight_variables(shape, identifier, trainable=True):
    """
    :param shape: shape of the weight matrice [num_rows,num_cols,depth]
    :param identifier: name of the weight matrice
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

    for matrice_index, matrice_neural_size in enumerate(fully_connected_sizes):
        with tf.name_scope(f"Fully-Connected-Layer-{matrice_index + 1}"):
            fully_connected_weights = weight_variables([model_outputs.get_shape()[-1], matrice_neural_size],
                                                       "Weights", trainable=True)
            fully_connected_bias = bias_variables([matrice_neural_size], "Biases", trainable=True)

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
