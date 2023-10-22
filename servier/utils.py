import os
import json
import yaml


def format_configuration_variables(config):
    """
    Formatting of configuration variables in a dictionary that can be passed to Trainer class, data preparation functions,
    project folder preparation function ...e.t.c.
    :param config: (dict) dictionary containing specified configurations that one would like to use
    :return:
    """
    project_configuration_variables = {
        "PROJECT_FOLDER": config["folder_structure"]["project_folder"],
        "train_csv_file": config["folder_structure"]["data_files"]["train"],
        "valid_csv_file": config["folder_structure"]["data_files"]["validation"],
        "test_csv_file": config["folder_structure"]["data_files"]["test"],
        "field_delimiter": config["data"]["field_delimiter"],
        "label_dictionary": config["data"]["label_dictionary"],
        "num_classes": len(config["data"]["label_dictionary"]),
        "use_finger_print": config["fingerprint"]["activated"],
        "finger_print_type": config["fingerprint"]["type"],
        "finger_print_parameters": config["fingerprint"]["parameters"][config["fingerprint"]["type"]] if
        config["fingerprint"]["type"] in ["morgan"] else {},
        "aggregation_type": config["model"]["aggregation"]["type"],
        "aggregation_parameters": config["model"]["aggregation"]["parameters"][
            config["model"]["aggregation"]["type"]] if config["model"]["aggregation"]["type"] else {},
        "fully_connected_sizes": config["model"]["fully_connected_sizes"],
        "weight_saver": config["training"]["settings"]["weight_saver"],
        "info_dump": config["training"]["settings"]["info_dump"],
        "num_iterations": config["training"]["settings"]["num_iterations"],
        "learning_rate": float(config["training"]["hyperparameters"]["learning_rate"]),
        "index_iteration": config["inference"]["index_iteration"],
        "scaler": config["inference"]["scaler"],
        "LOG_ERROR": config["logging"]["log_error"],
    }
    return project_configuration_variables


def load_configuration_variables(experiment_name="project_config.example.yml"):
    """
    Function that return project configuration variables based on the experiment file that is being used
    :param experiment_name: (str) experiment file name
    :return: project_configuration_variables : (dict) variable containing project configuration variables
    """

    # Get Current Folder as well as configuration file in the setup folder
    dental_mind_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    setup_file = os.path.join(dental_mind_folder, "config", experiment_name)

    # Load project configuration variables
    project_configuration_variables = format_configuration_variables(yaml.safe_load(open(setup_file, 'r')))

    return project_configuration_variables


def safe_dump(training_parameters, destination):
    """
    Function that dumps training parameters for traceability
    :param training_parameters: (dict) Dictionary containing training parameters
    :param destination: (str) Destination of the json file that will contain all the training parameters
    :return: Storage of params.json at destination path
    """
    try:
        with open(destination, "w") as fp:
            json.dump(
                training_parameters, fp, sort_keys=True, indent=4, separators=(",", ":")
            )
    except KeyboardInterrupt:
        safe_dump(training_parameters, destination)


def make_dir(path):
    """
    Function that creates a folder when there isn't one
    :param path: (str) Folder path that we want to create if non-existent
    :return: Folder creation if non-existent
    """
    if not (os.path.exists(path)):
        os.makedirs(path)


def print_blue(output):
    """
    :param output: string that we wish to print in a certain colour
    :return:
    """
    print("\033[94m" + "\033[1m" + output + "\033[0m")


def print_green(output):
    """
    :param output: string that we wish to print in a certain colour
    :return:
    """
    print("\033[32m" + "\033[1m" + output + "\033[0m")


def print_yellow(output):
    """
    :param output: string that we wish to print in a certain colour
    :return:
    """
    print("\033[93m" + "\033[1m" + output + "\033[0m")


def print_red(output):
    """
    :param output: string that we wish to print in a certain colour
    :return:
    """
    print("\033[91m" + "\033[1m" + output + "\033[0m")


def print_bold(output):
    """
    :param output: string that we wish to print in bold font
    :return:
    """
    print("\033[1m" + output + "\033[0m")
