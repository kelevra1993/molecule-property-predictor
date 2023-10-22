"""
File that contains code for the Trainer Class
"""
import os
import time
import tensorflow as tf
from utils import make_dir, print_green, safe_dump, print_yellow
from data_manager.data_utils import get_model_placeholders, create_input_producer, count_records
from models.model_utils import create_deep_learning_model, postprocess, setup_tensorboard, print_model_size, \
    restore_last_model


class Trainer:
    """
    Trainer class that can be used for training, evaluation and prediction
    """

    def __init__(self, **kwargs):

        self.PROJECT_FOLDER = kwargs.get("PROJECT_FOLDER")
        self.train_csv_file = kwargs.get("train_csv_file")
        self.valid_csv_file = kwargs.get("valid_csv_file")
        self.test_csv_file = kwargs.get("test_csv_file")
        self.field_delimiter = kwargs.get("field_delimiter")
        self.label_dictionary = kwargs.get("label_dictionary")
        self.num_classes = kwargs.get("num_classes")
        self.use_finger_print = kwargs.get("use_finger_print")
        self.finger_print_type = kwargs.get("finger_print_type")
        self.finger_print_parameters = kwargs.get("finger_print_parameters")
        self.aggregation_type = kwargs.get("aggregation_type")
        self.aggregation_parameters = kwargs.get("aggregation_parameters")
        self.fully_connected_sizes = kwargs.get("fully_connected_sizes")
        self.weight_saver = kwargs.get("weight_saver")
        self.info_dump = kwargs.get("info_dump")
        self.num_iterations = kwargs.get("num_iterations")
        self.learning_rate = kwargs.get("learning_rate")
        self.index_iteration = kwargs.get("index_iteration")
        self.scaler = kwargs.get("scaler")
        self.LOG_ERROR = kwargs.get("LOG_ERROR")
        self.training_variables = kwargs

        # Setting up csv column defaults
        # TODO label dictionary will give us the correct column_names
        self.column_defaults = [tf.float32, tf.string, tf.string]

        # Always initialise project paths containing model weights and results
        (self.raw_parameters, self.modelPath, self.ResultPath, self.weightPath,
         self.param_file, self.template_path, self.tensorboardPath) = self.initialize_project_paths()

        # Get input and output shapes
        self.sequence_input_shape, self.sequence_output_shape = self.get_data_input_output_shapes()

        # Set Up Session and Saver
        self.session = tf.compat.v1.InteractiveSession()

        # Setup input and output placeholders
        (self.sequence_data_placeholder, self.sequence_label_placeholder,
         self.normalization_phase) = get_model_placeholders(sequence_input_shape=self.sequence_input_shape,
                                                            sequence_output_shape=self.sequence_output_shape)
        # Create the deep learning model
        self.classification_tensor, self.softmax_tensor = create_deep_learning_model(
            inputs=self.sequence_data_placeholder,
            fully_connected_sizes=self.fully_connected_sizes,
            scaler=self.scaler,
            num_classes=self.num_classes)

        # Get outputs as well as optimizers
        self.training_step, self.cross_entropy_loss, self.accuracy = postprocess(
            classification=self.classification_tensor,
            label=self.sequence_label_placeholder,
            learning_rate=self.learning_rate)

        # Setup saver
        self.saver = tf.compat.v1.train.Saver(max_to_keep=None)

        print("Tensorflow Trainer - Evaluator Object Has Been Set")

    def initialize_project_paths(self):
        """
        Function that initializes project paths that indicate where models weights are results are stored
        :return: modelPath (str), ResultPath (str), weightPath (str),
         param_file (str), template_path (str), tensorboardPath (str)
        """
        # Raw parameters for model information storage
        params = ""
        if self.finger_print_type:

            if self.finger_print_type == "morgan":
                params += "MORGAN("
                params += f"{self.finger_print_parameters['radius']}-"
                params += f"{self.finger_print_parameters['size']}"
                params += "-CHIR" if self.finger_print_parameters['use_chirality'] else ""
                params += "-BT" if self.finger_print_parameters['use_bond_types'] else ""
                params += "-FEAT" if self.finger_print_parameters['use_features'] else ""
                params += ")-"

        # TODO ADD type of preprocessing that will be used ( either naive or expert systems )

        aggregation_name = ""
        if self.aggregation_type:
            aggregation_name = str(self.aggregation_type.upper()) + (
                "_" + "_".join([f"({value})" for value in self.aggregation_parameters.values()])
                if self.aggregation_type not in ["mean", "max"] else "") + "_"

        raw_parameters = (params + aggregation_name + '_'.join(map(str, self.fully_connected_sizes)))

        # Model path for Result Analysis, Error Analysis, Model Storing, Model Freezing if non-existant
        modelPath = os.path.join(self.PROJECT_FOLDER, "Models", raw_parameters)
        ResultPath = os.path.join(modelPath, "Results")
        weightPath = os.path.join(modelPath, "Weights")
        param_file = os.path.join(modelPath, "params.json")
        template_path = os.path.join("data_manager", "Template.xlsx")
        tensorboardPath = os.path.join(self.PROJECT_FOLDER, "Tensorboard")

        return raw_parameters, modelPath, ResultPath, weightPath, param_file, template_path, tensorboardPath

    def prepare_project_folder(self):
        """
        Function that create project folders in oder to launch training
        :return: result_evaluation_file (str) path to file containing training results
        """
        make_dir(self.weightPath)
        make_dir(self.ResultPath)
        make_dir(self.tensorboardPath)

        # Saving Training parameters to a "params.json" file if it does not exist
        if not (os.path.exists(self.param_file)):
            print_green("Creating params.json that contains Training and Validation Hyperparameters !!!\n")
            safe_dump(training_parameters=self.training_variables, destination=self.param_file)
        else:
            print_yellow("params.json already exists and does not need to be updated !!!\n")

        # Results of validation on saved Models
        result_evaluation_file = os.path.join(self.modelPath, "Results.txt")

        return result_evaluation_file

    def get_data_input_output_shapes(self):
        """
        Function that get data input and output shapes based on user configuration specifications
        :return: sequence_input_shape (list[int]),sequence_output_shape (list[int])
        """
        sequence_input_shape = None

        # Define sequence input shape and output shape
        if self.finger_print_type == "morgan":
            sequence_input_shape = [None, self.finger_print_parameters['size']]

        sequence_output_shape = [1, self.num_classes]

        return sequence_input_shape, sequence_output_shape

    def train(self):

        # Create project folders
        result_evaluation_file = self.prepare_project_folder()

        # Training Parameters
        train_initializer, training_data_tensor = create_input_producer(csv_file=self.train_csv_file,
                                                                        column_defaults=self.column_defaults,
                                                                        name_scope="Training-Batch",
                                                                        field_delimiter=self.field_delimiter)
        # Validation Parameters
        validation_initializer, validation_data_tensor = create_input_producer(csv_file=self.valid_csv_file,
                                                                               column_defaults=self.column_defaults,
                                                                               name_scope="Validation-Batch",
                                                                               field_delimiter=self.field_delimiter)
        # Testing Parameters
        test_initializer, test_data_tensor = create_input_producer(csv_file=self.test_csv_file,
                                                                   column_defaults=self.column_defaults,
                                                                   name_scope="Test-Batch",
                                                                   field_delimiter=self.field_delimiter)

        # Setting Up Tensorboard
        merged_view, train_file_writer, validation_file_writer = setup_tensorboard(loss=self.cross_entropy_loss,
                                                                                   accuracy=self.cross_entropy_loss,
                                                                                   session=self.session,
                                                                                   tensorboard_path=self.tensorboardPath)
        # Restore the last model used for training
        iteration_last_model = restore_last_model(path=self.weightPath,
                                                  session=self.session,
                                                  saver=self.saver,
                                                  last=True)

        # Initialize tensorflow variables and get number of element in testing database
        test_iterations = self.initialize_tensorflow_variables_and_print_summary()

        # Running our data pipeline initializers
        self.session.run([train_initializer, validation_initializer, test_initializer])

        # get trackers
        tracker_dictionary = self.get_trackers()

        print_green(30 * "-")
        print_green("Training Started....")
        print_green(30 * "-" + "\n")

        # Iterate through the model

    def get_trackers(self):
        """
        Function that gets elements that we would like to tack during training process
        :return:
        """

        tracker_dictionary = {"initial_start": time.time(),
                              "start": time.time(),
                              "training_moving_average": 0.0,
                              "validation_moving_average": 0.0,
                              "training_accuracy": 0.0,
                              "validation_accuracy": 0.0
                              }

        return tracker_dictionary

    def initialize_tensorflow_variables_and_print_summary(self):
        """
        Function that initializes tensorflow variables and prints out a summary
        :return: test_iterations (int) number of elements in our test dataset
        """
        self.session.run(tf.compat.v1.global_variables_initializer())
        self.session.run(tf.compat.v1.local_variables_initializer())

        # Fetch the number of smile sequences in our test dataset
        print_yellow("\nCounting Sequences In Our Test Dataset...")
        test_iterations = count_records(self.session, self.test_csv_file)
        print_green(f"There are {test_iterations} Smile String Sequences In Our Test Dataset")

        print(f"\nModel File : {self.raw_parameters}\n")
        print(f"Number Of Training Iterations : {self.num_iterations}")
        print(f"Number Of Validation Iterations : {test_iterations} \n")

        print_model_size()

        return test_iterations
