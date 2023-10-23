"""
File that contains code for the Trainer Class
"""
import os
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf

# Disable eager mode and set printer options
tf.compat.v1.disable_eager_execution()
np.set_printoptions(linewidth=200)
from utils import (make_dir, print_green, safe_dump, print_yellow,
                   print_blue, print_bold, print_red, plot_and_save_confusion_matrix)
from data_manager.data_utils import (get_model_placeholders, create_input_producer,
                                     count_records, prepare_data, dump_info, get_naive_encoder)
from models.model_utils import (create_deep_learning_model, postprocess, setup_tensorboard, print_model_size,
                                restore_last_model, console_log_update_tracker, manage_error_during_training,
                                dump_in_checkpoint)


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
        self.use_fingerprint = kwargs.get("use_fingerprint")
        self.fingerprint_type = kwargs.get("fingerprint_type")
        self.fingerprint_parameters = kwargs.get("fingerprint_parameters")
        self.aggregation_type = kwargs.get("aggregation_type")
        self.aggregation_parameters = kwargs.get("aggregation_parameters")
        self.fully_connected_sizes = kwargs.get("fully_connected_sizes")
        self.weight_saver = kwargs.get("weight_saver")
        self.info_dump = kwargs.get("info_dump")
        self.num_iterations = kwargs.get("num_iterations")
        self.learning_rate = kwargs.get("learning_rate")
        self.index_iteration = kwargs.get("index_iteration")
        self.scaler = kwargs.get("scaler")
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
            use_fingerprint=self.use_fingerprint,
            aggregation_type=self.aggregation_type,
            aggregation_parameters=self.aggregation_parameters,
            fully_connected_sizes=self.fully_connected_sizes,
            scaler=self.scaler,
            num_classes=self.num_classes)

        # Get outputs as well as optimizers
        self.training_step, self.cross_entropy_loss, self.accuracy_tensor = postprocess(
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
        if self.use_fingerprint and self.fingerprint_type:
            if self.fingerprint_type == "morgan":
                params += "MORGAN("
                params += f"{self.fingerprint_parameters['radius']}-"
                params += f"{self.fingerprint_parameters['size']}"
                params += "-CHIR" if self.fingerprint_parameters['use_chirality'] else ""
                params += "-BT" if self.fingerprint_parameters['use_bond_types'] else ""
                params += "-FEAT" if self.fingerprint_parameters['use_features'] else ""
                params += ")-"
        else:
            params += "NAIVE-"

        # TODO ADD type of preprocessing that will be used ( either naive or expert systems )

        aggregation_name = ""
        if self.aggregation_type and not self.use_fingerprint:
            aggregation_name = str(self.aggregation_type.upper()) + (
                    "_" + "_".join([f"({value})" for value in self.aggregation_parameters.values()]))
            aggregation_name+='-'

        raw_parameters = (params + aggregation_name + '_'.join(map(str, self.fully_connected_sizes)))

        # Model path for Result Analysis, Error Analysis, Model Storing, Model Freezing if non-existant
        modelPath = os.path.join(self.PROJECT_FOLDER, "Models", raw_parameters)
        ResultPath = os.path.join(modelPath, "Results")
        weightPath = os.path.join(modelPath, "Weights")
        param_file = os.path.join(modelPath, "params.json")
        template_path = os.path.join("servier", "data_manager", "Template.xlsx")
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
        if self.use_fingerprint and self.fingerprint_type == "morgan":
            sequence_input_shape = [None, self.fingerprint_parameters['size']]
        else:
            sequence_input_shape = [None, len(get_naive_encoder())]

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
        merged_summary_tensor, train_file_writer, validation_file_writer = setup_tensorboard(
            loss=self.cross_entropy_loss,
            accuracy=self.accuracy_tensor,
            session=self.session,
            tensorboard_path=self.tensorboardPath)

        # Initialize tensorflow variables and get number of element in testing database
        test_iterations = self.initialize_tensorflow_variables_and_print_summary()

        # Restore the last model used for training
        iteration_last_model = restore_last_model(path=self.weightPath,
                                                  session=self.session,
                                                  saver=self.saver,
                                                  last=True)

        # Running our data pipeline initializers
        self.session.run([train_initializer, validation_initializer, test_initializer])

        # get trackers
        tracker_dictionary = self.get_trackers()

        print_green(30 * "-")
        print_green("Training Started....")
        print_green(30 * "-" + "\n")

        for i in range(self.num_iterations):
            occ = iteration_last_model + i + 1

            try:

                training_data = self.session.run([training_data_tensor])[0]
                validation_data = self.session.run([validation_data_tensor])[0]

                # Get Training and validation data
                (sequence_label, sequence_id, sequence_name, sequence_data) = prepare_data(data=training_data,
                                                                                           num_classes=self.num_classes,
                                                                                           use_fingerprint=self.use_fingerprint,
                                                                                           fingerprint_type=self.fingerprint_type,
                                                                                           **self.fingerprint_parameters)

                (valid_sequence_label, valid_sequence_id, valid_sequence_name, valid_sequence_data) = prepare_data(
                    data=validation_data,
                    num_classes=self.num_classes,
                    use_fingerprint=self.use_fingerprint,
                    fingerprint_type=self.fingerprint_type,
                    **self.fingerprint_parameters)

                # Launch one forward and backward pass
                (_, training_loss, training_softmax, training_accuracy, training_summary,) = self.session.run(
                    [self.training_step, self.cross_entropy_loss, self.softmax_tensor, self.accuracy_tensor,
                     merged_summary_tensor],
                    feed_dict={self.sequence_data_placeholder: sequence_data,
                               self.sequence_label_placeholder: [sequence_label]})

                # Run the Neural Network for Validation
                (validation_loss, validation_softmax, validation_accuracy, validation_summary,) = self.session.run(
                    [self.cross_entropy_loss, self.softmax_tensor, self.accuracy_tensor, merged_summary_tensor],
                    feed_dict={self.sequence_data_placeholder: valid_sequence_data,
                               self.sequence_label_placeholder: [valid_sequence_label]})

                # Update trackers
                tracker_dictionary["training_moving_loss"] += training_loss
                tracker_dictionary["training_moving_accuracy"] += training_accuracy

                tracker_dictionary["validation_moving_loss"] += validation_loss
                tracker_dictionary["validation_moving_accuracy"] += validation_accuracy

                if occ % self.info_dump == 0:
                    # Dump information to console log
                    tracker_dictionary = console_log_update_tracker(iterations=occ,
                                                                    tracker_dictionary=tracker_dictionary,
                                                                    info_dump=self.info_dump)
                # Launching evaluation on test dataset
                if occ % self.weight_saver == 0:
                    self.launch_evaluation(
                        iteration=occ,
                        test_iterations=test_iterations,
                        test_data_tensor=test_data_tensor,
                        results_folder_path=self.ResultPath,
                        inference_during_training=True,
                        result_evaluation_file=result_evaluation_file)

                    print_yellow("Model is being saved.....")
                    self.saver.save(self.session, self.weightPath + "/Iteration_%d" % occ)
                    dump_in_checkpoint(self.weightPath, occ)
                    print_green("Model has been saved successfully\n\n")

                # Dumping Information Into Tensorboard
                train_file_writer.add_summary(training_summary, occ)
                validation_file_writer.add_summary(validation_summary, occ)

            except KeyboardInterrupt:

                manage_error_during_training(iteration=occ, message="\nTraining Was Abruptly Interrupted",
                                             saver=self.saver, session=self.session, weight_path=self.weightPath)

            except:
                raise
                manage_error_during_training(iteration=occ, message="\nUnknown Error During Training",
                                             saver=self.saver, session=self.session, weight_path=self.weightPath)

    def launch_evaluation(self, iteration, test_iterations, test_data_tensor, results_folder_path,
                          inference_during_training=False, result_evaluation_file=None):
        """
        Function that launches evaluation on the test set or any given set during training and evaluation phase
        :param iteration: (int) iteration of the model that is of interest
        :param test_iterations: (int) number of test iterations
        :param test_data_tensor: (Tensor) test data pipeline tensor
        :param results_folder_path: (str) path to result folder
        :param inference_during_training: (bool) boolean that indicates that we are in the training phase
        :param result_evaluation_file:  (str) file where we store metrics during training
        :return:
        """
        # Counter Initiation
        (correctly_predicted_dictionary, counter_dictionary, data_dictionary,
         confusion_matrix) = self.initialize_counters(label_dictionary=self.label_dictionary,
                                                      num_classes=self.num_classes)

        # Test accuracy on test set
        test_accuracy = 0.0
        start_test_timer = time.time()
        print(f"\nWe Are At Iteration {iteration} , Running Neural Network On The Whole Testing Dataset\n")

        time.sleep(1)

        for test_iteration in tqdm(range(test_iterations), desc="Model Evaluation"):
            test_data = self.session.run([test_data_tensor])[0]

            (sequence_label, sequence_id, sequence_name, sequence_data) = prepare_data(data=test_data,
                                                                                       num_classes=self.num_classes,
                                                                                       use_fingerprint=self.use_fingerprint,
                                                                                       fingerprint_type=self.fingerprint_type,
                                                                                       **self.fingerprint_parameters)
            # launch inference
            softmax, accuracy = self.session.run([self.softmax_tensor, self.accuracy_tensor],
                                                 feed_dict={self.sequence_data_placeholder: sequence_data,
                                                            self.sequence_label_placeholder: [sequence_label]})

            test_accuracy += accuracy

            # Fine grained evaluation
            # Dealing with a given label
            label_index = np.argmax(sequence_label)
            predicted_index = np.argmax(softmax[0], 0)
            counter_dictionary[self.label_dictionary[label_index]] += 1

            # Add element to confusion matrix
            confusion_matrix[predicted_index][label_index] += 1

            # A Correct prediction
            if predicted_index == label_index:
                data_dictionary[self.label_dictionary[label_index]].append(
                    (sequence_name, "Right", softmax[0][predicted_index]))
                correctly_predicted_dictionary[self.label_dictionary[label_index]] += 1

            # A False prediction
            else:
                data_dictionary[self.label_dictionary[predicted_index]].append(
                    (sequence_name, "False", softmax[0][predicted_index]))

        time.sleep(1)
        test_accuracy = 100 * (test_accuracy / test_iterations)

        message_list = []
        message_list.append("\n" + 50 * "-")
        message_list.append(f"Model Iteration {iteration} Global Accuracy : {np.round(test_accuracy, 2)}%")
        class_recalls = []
        for k, v in correctly_predicted_dictionary.items():
            message_list.append(f"Recall On {k} Class Is : {np.round(100 * (v / counter_dictionary[k]), 2)}%")
            class_recalls.append(100 * v / counter_dictionary[k])
        message_list.append(f"Average Recall On Both Classes Is {np.round(np.mean(class_recalls), 2)}%")
        message_list.append(50 * "-")

        if inference_during_training:
            target = open(result_evaluation_file, "a")
            for message in message_list:
                target.write(message + '\n')
            target.close()

        for message in message_list:
            print_blue(message) if "Recall" in message else print_bold(message)

        print(f"The Test Process Took {int((time.time() - start_test_timer))} seconds\n")

        iteration_result_folder = os.path.join(results_folder_path, f"Iteration_{iteration}")
        make_dir(iteration_result_folder)

        # Then we dump information to a text file
        dump_info(self.label_dictionary,
                  data_dictionary=data_dictionary,
                  counter_dictionary=counter_dictionary,
                  template_path=self.template_path,
                  output_file=os.path.join(iteration_result_folder, f"Results_{iteration}.xlsx"),
                  scaler=self.scaler)

        try:
            # Plot And Save Confusion Matrix
            plot_and_save_confusion_matrix(
                label_dictionary=self.label_dictionary,
                num_classes=self.num_classes,
                confusion_matrix=confusion_matrix,
                model_iteration=iteration,
                iteration_result_folder=iteration_result_folder)
        except:
            raise

    def get_trackers(self):
        """
        Function that gets elements that we would like to tack during training process
        :return:
        """

        tracker_dictionary = {"initial_start": time.time(),
                              "start": time.time(),
                              "training_moving_accuracy": 0.0,
                              "validation_moving_accuracy": 0.0,
                              "training_moving_loss": 0.0,
                              "validation_moving_loss": 0.0}

        return tracker_dictionary

    def initialize_counters(self, label_dictionary, num_classes):
        """
        # Setting dictionaries that keep track of classification prediction for metric computations and displays
        :param label_dictionary: dictionary containing the names of the desired classes
        :param num_classes: number of classes that are being predicted
        :return:
        """
        counter_dictionary = {}
        correctly_predicted_dictionary = {}
        data_dictionary = {}
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

        for i in range(len(label_dictionary)):
            counter_dictionary[label_dictionary[i]] = 0
            correctly_predicted_dictionary[label_dictionary[i]] = 0
            data_dictionary[label_dictionary[i]] = []

        return correctly_predicted_dictionary, counter_dictionary, data_dictionary, confusion_matrix

    def initialize_tensorflow_variables_and_print_summary(self):
        """
        Function that initializes tensorflow variables and prints out a summary
        :return: test_iterations (int) number of elements in our test dataset
        """
        self.session.run(tf.compat.v1.global_variables_initializer())
        self.session.run(tf.compat.v1.local_variables_initializer())

        # Fetch the number of smile sequences in our test dataset
        print_yellow("\nCounting Sequences In Our Test Dataset...")
        test_iterations = count_records(session=self.session, column_defaults=self.column_defaults,
                                        csv_file=self.test_csv_file, field_delimiter=self.field_delimiter)
        print_green(f"There are {test_iterations} Smile String Sequences In Our Test Dataset")

        print(f"\nModel File : {self.raw_parameters}\n")
        print(f"Number Of Training Iterations : {self.num_iterations}")
        print(f"Number Of Validation Iterations : {test_iterations} \n")

        print_model_size()

        return test_iterations
