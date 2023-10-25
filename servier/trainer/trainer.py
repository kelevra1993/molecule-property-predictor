"""
File that contains code for the Trainer Class
"""
import os
import time
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from utils import (make_dir, print_green, safe_dump, print_yellow, print_red,
                   print_blue, print_bold, plot_and_save_confusion_matrix)
from data_manager.data_utils import (get_model_placeholders, create_input_producer, preprocess_string,
                                     count_records, prepare_data, dump_info, get_naive_encoder)
from models.model_utils import (create_deep_learning_model, postprocess, setup_tensorboard, print_model_size,
                                restore_last_model, console_log_update_tracker, manage_error_during_training,
                                dump_in_checkpoint)

# Disable eager mode and set printer options
tf.compat.v1.disable_eager_execution()
np.set_printoptions(linewidth=200)


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
        self.multiple_property_prediction = kwargs.get("multiple_property_prediction")
        self.number_of_prediction_columns = kwargs.get("number_of_prediction_columns")
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
        self.training_variables = kwargs

        # Setting up csv column defaults, that define types for each column
        self.column_defaults = self.get_column_defaults()

        # Always initialise project paths containing model weights and results
        (self.raw_parameters, self.model_path, self.result_path, self.weight_path,
         self.param_file, self.template_path, self.tensorboard_path) = self.initialize_project_paths()

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
            multiple_property_prediction=self.multiple_property_prediction,
            number_of_prediction_columns=self.number_of_prediction_columns,
            aggregation_type=self.aggregation_type,
            aggregation_parameters=self.aggregation_parameters,
            fully_connected_sizes=self.fully_connected_sizes,
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
        :return: model_path (str), result_path (str), weight_path (str),
         param_file (str), template_path (str), tensorboard_path (str)
        """
        # Raw parameters for model information storage
        params = ""

        # Dealing with a model that predicts multiple properties
        if self.multiple_property_prediction and self.number_of_prediction_columns:
            params += f"MULTI-{self.number_of_prediction_columns}-"

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

        if not self.use_fingerprint and not self.aggregation_type:
            print_red("Since you are not using fingerprint, please specify the type of aggregator to use")
            print_red("This is done in your configuration file under aggregation > type ")
            exit()

        aggregation_name = ""
        if self.aggregation_type and not self.use_fingerprint:
            aggregation_name = str(self.aggregation_type.upper()) + (
                    "_" + "_".join([f"({value})" for value in self.aggregation_parameters.values()]))
            aggregation_name += '-'

        raw_parameters = (params + aggregation_name + '_'.join(map(str, self.fully_connected_sizes)))

        # Model path for Result Analysis, Error Analysis, Model Storing, Model Freezing if non-existant
        model_path = os.path.join(self.PROJECT_FOLDER, "Models", raw_parameters)
        result_path = os.path.join(model_path, "Results")
        weight_path = os.path.join(model_path, "Weights")
        param_file = os.path.join(model_path, "params.json")
        template_path = os.path.join("servier", "data_manager", "Template.xlsx")
        tensorboard_path = os.path.join(self.PROJECT_FOLDER, "Tensorboard")

        return raw_parameters, model_path, result_path, weight_path, param_file, template_path, tensorboard_path

    def prepare_project_folder(self):
        """
        Function that create project folders in oder to launch training
        :return: result_evaluation_file (str) path to file containing training results
        """
        make_dir(self.weight_path)
        make_dir(self.result_path)
        make_dir(self.tensorboard_path)

        # Saving Training parameters to a "params.json" file if it does not exist
        if not (os.path.exists(self.param_file)):
            print_green("Creating params.json that contains Training and Validation Hyperparameters !!!\n")
            safe_dump(training_parameters=self.training_variables, destination=self.param_file)
        else:
            print_yellow("params.json already exists and does not need to be updated !!!\n")

        # Results of validation on saved Models
        result_evaluation_file = os.path.join(self.model_path, "Results.txt")

        return result_evaluation_file

    def get_column_defaults(self):
        """
        Function that gets column defaults for propery prediction
        :return: List(Tensor) [tensor_type_property, tensor_type_id, tensor_type_smile_string e.t.c]
        """
        # Setting up csv column defaults
        if self.multiple_property_prediction:
            column_defaults = []
            for i in range(self.number_of_prediction_columns):
                column_defaults.append(tf.float32)
            # Add sequence id as well as smile string
            column_defaults.extend([tf.string, tf.string])
        else:
            column_defaults = [tf.float32, tf.string, tf.string]
        return column_defaults

    def get_data_input_output_shapes(self):
        """
        Function that get data input and output shapes based on user configuration specifications
        :return: sequence_input_shape (list[int]),sequence_output_shape (list[int])
        """

        # Define sequence input shape and output shape
        if self.use_fingerprint and self.fingerprint_type == "morgan":
            sequence_input_shape = [None, self.fingerprint_parameters['size']]
        else:
            sequence_input_shape = [None, len(get_naive_encoder())]

        # Dealing with multi-classification
        if self.multiple_property_prediction:
            sequence_output_shape = [self.number_of_prediction_columns, self.num_classes]
        else:
            sequence_output_shape = [1, self.num_classes]

        return sequence_input_shape, sequence_output_shape

    def get_training_input_producers(self):
        """
        Function that sets up data input pipelines for training validation and testing
        :return:
        """
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

        return (train_initializer, training_data_tensor, validation_initializer,
                validation_data_tensor, test_initializer, test_data_tensor)

    def train(self):

        # Create project folders
        result_evaluation_file = self.prepare_project_folder()

        # Get Training, Validation and Test Parameters
        (train_initializer, training_data_tensor, validation_initializer, validation_data_tensor, test_initializer,
         test_data_tensor) = self.get_training_input_producers()

        # Setting Up Tensorboard
        merged_summary_tensor, train_file_writer, validation_file_writer = setup_tensorboard(
            loss=self.cross_entropy_loss,
            accuracy=self.accuracy_tensor,
            session=self.session,
            tensorboard_path=self.tensorboard_path)

        # Initialize tensorflow variables and get number of element in testing database
        test_iterations = self.initialize_tensorflow_variables_and_print_summary()

        # Restore the last model used for training
        iteration_last_model = restore_last_model(path=self.weight_path,
                                                  session=self.session,
                                                  saver=self.saver)

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
                # Get training and validation un-formatted data for input pipelines
                training_data = self.session.run([training_data_tensor])[0]
                validation_data = self.session.run([validation_data_tensor])[0]

                # Get Training and validation data
                (sequence_label, sequence_id, sequence_name, sequence_data) = prepare_data(
                    data=training_data,
                    num_classes=self.num_classes,
                    multiple_property_prediction=self.multiple_property_prediction,
                    number_of_prediction_columns=self.number_of_prediction_columns,
                    use_fingerprint=self.use_fingerprint,
                    fingerprint_type=self.fingerprint_type,
                    **self.fingerprint_parameters)

                (valid_sequence_label, valid_sequence_id, valid_sequence_name, valid_sequence_data) = prepare_data(
                    data=validation_data,
                    num_classes=self.num_classes,
                    multiple_property_prediction=self.multiple_property_prediction,
                    number_of_prediction_columns=self.number_of_prediction_columns,
                    use_fingerprint=self.use_fingerprint,
                    fingerprint_type=self.fingerprint_type,
                    **self.fingerprint_parameters)

                # Launch one forward and backward pass
                (_, training_loss, training_softmax, training_accuracy, training_summary,) = self.session.run(
                    [self.training_step, self.cross_entropy_loss, self.softmax_tensor, self.accuracy_tensor,
                     merged_summary_tensor],
                    feed_dict={self.sequence_data_placeholder: sequence_data,
                               self.sequence_label_placeholder: [
                                   sequence_label] if not self.multiple_property_prediction else sequence_label})

                # Run the Neural Network for Validation
                (validation_loss, validation_softmax, validation_accuracy, validation_summary,) = self.session.run(
                    [self.cross_entropy_loss, self.softmax_tensor, self.accuracy_tensor, merged_summary_tensor],
                    feed_dict={self.sequence_data_placeholder: valid_sequence_data,
                               self.sequence_label_placeholder: [
                                   valid_sequence_label] if not self.multiple_property_prediction else valid_sequence_label})

                # Update loss and accuracy trackers for training and validation
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
                    self.launch_evaluation_on_single_model(
                        iteration=occ,
                        test_iterations=test_iterations,
                        test_data_tensor=test_data_tensor,
                        results_folder_path=self.result_path,
                        inference_during_training=True,
                        result_evaluation_file=result_evaluation_file)

                    print_yellow("Model is being saved.....")
                    self.saver.save(self.session, self.weight_path + "/Iteration_%d" % occ)
                    dump_in_checkpoint(self.weight_path, occ)
                    print_green("Model has been saved successfully\n\n")

                # Dumping Information Into Tensorboard
                train_file_writer.add_summary(training_summary, occ)
                validation_file_writer.add_summary(validation_summary, occ)

            except KeyboardInterrupt:
                # Save the model then stop the script
                manage_error_during_training(iteration=occ, message="\nTraining Was Abruptly Interrupted",
                                             saver=self.saver, session=self.session, weight_path=self.weight_path)

            except:
                # Save the model then stop the script
                manage_error_during_training(iteration=occ, message="\nUnknown Error During Training",
                                             saver=self.saver, session=self.session, weight_path=self.weight_path)
                raise

    def evaluate(self, iteration):
        """
        Function that is used to evaluate a model given a testing database.
        :param iteration: (int) iteration that was saved during training and that is of interest to us.
        :return:
        """
        # Evaluation Data Pipeline Parameters
        test_initializer, test_data_tensor = create_input_producer(csv_file=self.test_csv_file,
                                                                   column_defaults=self.column_defaults,
                                                                   name_scope="Test-Batch",
                                                                   field_delimiter=self.field_delimiter)

        # Initialize tensorflow variables and get number of element in testing database
        test_iterations = self.initialize_tensorflow_variables_and_print_summary(training=False)

        # Restore the specific model that interests us the most
        _ = restore_last_model(path=self.weight_path,
                               session=self.session,
                               saver=self.saver,
                               index_iteration=iteration)

        # Running our data pipeline initializer
        self.session.run([test_initializer])

        # Set Result Folder path for evaluation
        self.result_path = os.path.join(os.path.dirname(self.result_path),
                                        f"Evaluation-{os.path.basename(self.result_path)}")
        make_dir(self.result_path)

        self.launch_evaluation_on_single_model(
            iteration=iteration,
            test_iterations=test_iterations,
            test_data_tensor=test_data_tensor,
            results_folder_path=self.result_path,
            inference_during_training=False,
            result_evaluation_file=None)

    def predict(self, iteration, smile_string):

        # initializing prediction variables
        self.session.run(tf.compat.v1.global_variables_initializer())
        self.session.run(tf.compat.v1.local_variables_initializer())

        # Restore the specific model that interests us the most
        _ = restore_last_model(path=self.weight_path,
                               session=self.session,
                               saver=self.saver,
                               index_iteration=iteration)

        # Preprocess Data
        sequence_data = preprocess_string(
            use_fingerprint=self.use_fingerprint,
            fingerprint_type=self.fingerprint_type,
            smile_string=smile_string,
            **self.fingerprint_parameters)

        # Run Inference
        # launch inference for the smile string
        softmax = self.session.run(self.softmax_tensor, feed_dict={self.sequence_data_placeholder: sequence_data
                                                                   })

        if self.multiple_property_prediction:

            prediction_dictionary = {}
            for sample_index, sample_column in enumerate(["P2", "P1", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]):
                predicted_index = np.argmax(softmax[sample_index], 0)
                prediction_dictionary[sample_column] = self.label_dictionary[predicted_index]
            return prediction_dictionary
        else:
            predicted_index = np.argmax(softmax[0], 0)
            return {"P1": self.label_dictionary[predicted_index]}

    def launch_evaluation_on_single_model(self, iteration, test_iterations, test_data_tensor, results_folder_path,
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
        (correctly_predicted_dictionary, multi_correctly_predicted_dictionary, counter_dictionary,
         multi_counter_dictionary, data_dictionary, confusion_matrix) = self.initialize_counters(
            label_dictionary=self.label_dictionary,
            num_classes=self.num_classes)

        # Test accuracy on test set
        test_accuracy = 0.0
        start_test_timer = time.time()
        print(f"\nWe Are At Iteration {iteration} , Running Neural Network On The Whole Testing Dataset\n")

        time.sleep(1)

        for _ in tqdm(range(test_iterations), desc="Model Evaluation"):
            test_data = self.session.run([test_data_tensor])[0]

            (sequence_label, sequence_id, sequence_name, sequence_data) = prepare_data(
                data=test_data,
                num_classes=self.num_classes,
                multiple_property_prediction=self.multiple_property_prediction,
                number_of_prediction_columns=self.number_of_prediction_columns,
                use_fingerprint=self.use_fingerprint,
                fingerprint_type=self.fingerprint_type,
                **self.fingerprint_parameters)

            # launch inference
            softmax, accuracy = self.session.run([self.softmax_tensor, self.accuracy_tensor],
                                                 feed_dict={self.sequence_data_placeholder: sequence_data,
                                                            self.sequence_label_placeholder: [
                                                                sequence_label] if not self.multiple_property_prediction else sequence_label})

            test_accuracy += accuracy

            # Updating counter for multi-property predictions as well as single property prediction
            if self.multiple_property_prediction:
                multi_counter_dictionary, multi_correctly_predicted_dictionary = self.update_multi_counters(
                    multi_counter_dictionary=multi_counter_dictionary,
                    multi_correctly_predicted_dictionary=multi_correctly_predicted_dictionary,
                    softmax=softmax,
                    sequence_label=sequence_label)
                continue
            else:
                (counter_dictionary, correctly_predicted_dictionary,
                 confusion_matrix, data_dictionary) = self.update_single_counters(
                    counter_dictionary=counter_dictionary,
                    correctly_predicted_dictionary=correctly_predicted_dictionary,
                    confusion_matrix=confusion_matrix,
                    data_dictionary=data_dictionary,
                    sequence_name=sequence_name,
                    sequence_label=sequence_label,
                    softmax=softmax)

        time.sleep(1)
        test_accuracy = 100 * (test_accuracy / test_iterations)

        # Get list of messages that will be displayed to the user containing recalls per class
        # as well as averaged recall.
        message_list = self.get_accuracy_and_recall_messages(
            iteration=iteration,
            test_accuracy=test_accuracy,
            counter_dictionary=counter_dictionary,
            multi_counter_dictionary=multi_counter_dictionary,
            correctly_predicted_dictionary=correctly_predicted_dictionary,
            multi_correctly_predicted_dictionary=multi_correctly_predicted_dictionary)

        # Only write to result file during training
        if inference_during_training:
            target = open(result_evaluation_file, "a")
            for message in message_list:
                target.write(message + '\n')
            target.close()

        for message in message_list:
            print_blue(message) if "Recall" in message else print_bold(message)

        print(f"The Test Process Took {int((time.time() - start_test_timer))} seconds\n")

        # Multi-property-prediction information dump has not been implemented
        # therefore no need to continue
        if self.multiple_property_prediction:
            return None

        # Create folder that will store results for a particular iteration
        iteration_result_folder = os.path.join(results_folder_path, f"Iteration_{iteration}")
        make_dir(iteration_result_folder)

        # Then we dump information to a text file
        dump_info(self.label_dictionary,
                  data_dictionary=data_dictionary,
                  counter_dictionary=counter_dictionary,
                  template_path=self.template_path,
                  output_file=os.path.join(iteration_result_folder, f"Results_{iteration}.xlsx"))

        # Plot And Save Confusion Matrix
        plot_and_save_confusion_matrix(
            label_dictionary=self.label_dictionary,
            num_classes=self.num_classes,
            confusion_matrix=confusion_matrix,
            model_iteration=iteration,
            iteration_result_folder=iteration_result_folder)

    @staticmethod
    def get_trackers():
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

        multi_counter_dictionary = {}
        multi_correctly_predicted_dictionary = {}
        if self.multiple_property_prediction:
            for i in range(1, self.number_of_prediction_columns + 1):
                multi_counter_dictionary[f"P{i}"] = {v: 0 for v in label_dictionary.values()}
                multi_correctly_predicted_dictionary[f"P{i}"] = {v: 0 for v in label_dictionary.values()}

        return (correctly_predicted_dictionary, multi_correctly_predicted_dictionary, counter_dictionary,
                multi_counter_dictionary, data_dictionary, confusion_matrix)

    def update_single_counters(self, counter_dictionary, correctly_predicted_dictionary, confusion_matrix,
                               data_dictionary, sequence_name, sequence_label, softmax):
        """

        :param counter_dictionary: (dict) dictionary containing counts of elements per class
        :param correctly_predicted_dictionary: (dict) dictionary containing counts of correct prediction elements per class
        :param confusion_matrix: (dict) dictionary containing data that is used to create the confusion matrix
        :param data_dictionary: (dict) dictionary containing detailed predictions
        :param sequence_name: (str) sequence name string
        :param sequence_label: [float,float] labels
        :param softmax: [float,float] softmax applied on predictions
        :return:
        """

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

        return counter_dictionary, correctly_predicted_dictionary, confusion_matrix, data_dictionary

    def update_multi_counters(self, multi_counter_dictionary, multi_correctly_predicted_dictionary, softmax,
                              sequence_label):
        """
        Function that updates counters that track correct predictions and total sample for multi-property predictions
        :param multi_counter_dictionary: (dict) dictionary containing number of samples per property
        :param multi_correctly_predicted_dictionary: (dict) dictionary containing correct predictions per property
        :param softmax: List(float,float) list of tuple for softmax applied on each property prediction
        :param sequence_label: List(float,float) sequence labels for each property
        :return:
        """
        # For each column prediction count the elements as well as keep track of correct predictions
        # Temporary, ignoring moving forward since nothing is yet coded for display
        for sample_index, sample_column in enumerate(["P2", "P1", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]):
            label_index = np.argmax(sequence_label[sample_index])
            predicted_index = np.argmax(softmax[sample_index], 0)
            multi_counter_dictionary[sample_column][self.label_dictionary[label_index]] += 1
            if predicted_index == label_index:
                multi_correctly_predicted_dictionary[sample_column][self.label_dictionary[label_index]] += 1

        return multi_counter_dictionary, multi_correctly_predicted_dictionary

    def get_accuracy_and_recall_messages(self, iteration, test_accuracy, counter_dictionary, multi_counter_dictionary,
                                         correctly_predicted_dictionary, multi_correctly_predicted_dictionary):
        """

        :param iteration: (int) iteration of the model that we are currently looking at
        :param test_accuracy: (float) accuracy evaluated on the whole test dataset
        :param counter_dictionary:(dict) dictionary containing number of samples per P1 property
        :param correctly_predicted_dictionary: (dict) dictionary containing correct predictions per P1 property
        :param multi_counter_dictionary: (dict) dictionary containing number of samples per property
        :param multi_correctly_predicted_dictionary: (dict) dictionary containing correct predictions per property

        :return:
        """
        message_list = ["\n" + 50 * "-", f"Model Iteration {iteration} Global Accuracy : {np.round(test_accuracy, 2)}%"]
        class_recalls = []

        if self.multiple_property_prediction:
            for prop, true_positive_property in multi_correctly_predicted_dictionary.items():
                property_class_recall = []
                for k, v in true_positive_property.items():
                    message_list.append(
                        f"Recall On {k} Class For {prop} Is : {np.round(100 * (v / multi_counter_dictionary[prop][k]), 2)}%")
                    property_class_recall.append(100 * v / multi_counter_dictionary[prop][k])
                message_list.append(f"Average Recall On Both Classes Is {np.round(np.mean(property_class_recall), 2)}%")
                message_list.append(str(20 * "-") + f"{prop}" + str(20 * "-"))
            message_list.append(50 * "-")
        else:
            for k, v in correctly_predicted_dictionary.items():
                message_list.append(f"Recall On {k} Class Is : {np.round(100 * (v / counter_dictionary[k]), 2)}%")
                class_recalls.append(100 * v / counter_dictionary[k])
            message_list.append(f"Average Recall On Both Classes Is {np.round(np.mean(class_recalls), 2)}%")
            message_list.append(50 * "-")

        return message_list

    def initialize_tensorflow_variables_and_print_summary(self, training=True):
        """
        Function that initializes tensorflow variables and prints out a summary
        :param: training (boolean) variable that indicates that we are in the training phase
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
        if training:
            print(f"Number Of Training Iterations : {self.num_iterations}")
        print(f"Number Of Validation Iterations : {test_iterations} \n")

        print_model_size()

        return test_iterations
