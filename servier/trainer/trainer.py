"""
File that contains code for the Trainer Class
"""
import os
import tensorflow as tf
from utils import make_dir, print_green, safe_dump, print_yellow


class Trainer:
    """
    Trainer class that can be used for training, evaluation and prediction
    """

    def __init__(self, **kwargs):

        self.PROJECT_FOLDER = kwargs.get("PROJECT_FOLDER")
        self.train_csv_file = kwargs.get("train_csv_file")
        self.valid_csv_file = kwargs.get("valid_csv_file")
        self.field_delimiter = kwargs.get("field_delimiter")
        self.label_dictionary = kwargs.get("label_dictionary")
        self.num_classes = kwargs.get("num_classes")
        self.use_finger_print = kwargs.get("use_finger_print")
        self.finger_print_type = kwargs.get("finger_print_type")
        self.finger_print_parameters = kwargs.get("finger_print_parameters")
        self.aggregation_type = kwargs.get("aggregation_type")
        self.aggregation_parameters = kwargs.get("aggregation_parameters")
        self.fully_connected_sizes = kwargs.get("fully_connected_sizes")
        self.tensorboard = kwargs.get("tensorboard")
        self.weight_saver = kwargs.get("weight_saver")
        self.info_dump = kwargs.get("info_dump")
        self.num_iterations = kwargs.get("num_iterations")
        self.learning_rate = kwargs.get("learning_rate")
        self.index_iteration = kwargs.get("index_iteration")
        self.scaler = kwargs.get("scaler")
        self.LOG_ERROR = kwargs.get("LOG_ERROR")
        self.training_variables = kwargs

        # Always initialise project paths containing model weights and results
        (self.modelPath, self.ResultPath, self.weightPath,
         self.param_file, self.template_path, self.tensorboardPath) = self.initialize_project_paths()

        # Get input and output shapes
        self.sequence_input_shape, self.sequence_output_shape = self.get_data_input_output_shapes()

        self.session = tf.compat.v1.InteractiveSession()

        print_green("project is set")

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

        return modelPath, ResultPath, weightPath, param_file, template_path, tensorboardPath

    def prepare_project_folder(self):
        """
        Function that create project folders in oder to launch training
        :return: result_evaluation_file (str) path to file containing training results
        """
        make_dir(self.weightPath)
        make_dir(self.ResultPath)

        # Saving Training parameters to a "params.json" file if it does not exist
        if not (os.path.exists(self.param_file)):
            print_green("Creating params.json that contains Training and Validation Hyperparameters !!!\n")
            safe_dump(training_parameters=self.training_variables, destination=self.param_file)
        else:
            print_yellow("params.json already exists and does not need to be updated !!!\n")

        # Tensorboard Path
        if self.tensorboard:
            make_dir(self.tensorboardPath)

        # TODO TO BE RECODED, WE MIGHT USE A CSV INSTEAD
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

    def set_up_model(self):
        pass

    def train(self):

        # Create project folders
        result_evaluation_file = self.prepare_project_folder()

        # TODO label dictionary will give us the correct column_names
        column_defaults = [tf.float32, tf.string, tf.string]

