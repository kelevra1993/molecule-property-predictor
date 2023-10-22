import argparse

from utils import load_configuration_variables, print_red
from trainer.trainer import Trainer


def main(configuration_file, train_mode, evaluate_mode, predict_mode):
    """
    Function that manages either training, evaluation or prediction depending on user input arguments
    :param configuration_file: (str) configuration file containing experiments
    :param train_mode: (boolean) indicates that we want to train a model
    :param evaluate_mode: (boolean) indicates that we want to evaluate a model
    :param predict_mode: (boolean) indicates that we want to predict a smile sequences properties
    :return:
    """
    project_configuration_variables = load_configuration_variables(experiment_name=configuration_file)

    # Feed Variables To Trainer Object
    trainer_object = Trainer(**project_configuration_variables)

    if train_mode:
        # do training
        pass

    if evaluate_mode:
        # do evaluation
        pass

    if predict_mode:
        # do prediction
        pass

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Getting project parameters
    parser.add_argument("--config", type=str, default='project_config.example.yml',
                        help='name of configuration file that will be used')
    parser.add_argument('--train', action='store_true',
                        help='Boolean that indicates that we are launching training')
    parser.add_argument('--evaluate', action='store_true',
                        help='Boolean that indicates that we are launching evaluation')
    parser.add_argument('--predict', action='store_true',
                        help='Boolean that indicates that we are launching prediction')

    parsed_arguments, un_parsed_arguments = parser.parse_known_args()

    # Make sure that arguments are coherent
    # At least one mode has to be True, either train, evaluate or predict and only one mode has to be True
    if not (parsed_arguments.train or parsed_arguments.evaluate or parsed_arguments.predict):
        print_red("Please Specify The Mode That You Would Like To Launch, either train, evaluate or predict")
        exit()
    if [parsed_arguments.train, parsed_arguments.evaluate, parsed_arguments.predict].count(True) > 1:
        print_red("Please Specify Only One Mode, either train, evaluate or predict")
        exit()

    # Run the main function
    main(configuration_file=parsed_arguments.config,
         train_mode=parsed_arguments.train,
         evaluate_mode=parsed_arguments.evaluate,
         predict_mode=parsed_arguments.predict)
