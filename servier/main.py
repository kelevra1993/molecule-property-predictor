import os
import argparse
from utils import load_configuration_variables, print_red, print_green
from trainer.trainer import Trainer


def main(configuration_file, train_mode, evaluate_mode, predict_mode, model_iteration, smile_string):
    """
    Function that manages either training, evaluation or prediction depending on user input arguments
    :param configuration_file: (str) configuration file containing experiments
    :param train_mode: (boolean) indicates that we want to train a model
    :param evaluate_mode: (boolean) indicates that we want to evaluate a model
    :param predict_mode: (boolean) indicates that we want to predict a smile sequences properties
    :param model_iteration: (int) specific iteration that is of interest to the user
    :param smile_string: (str) molecule smile string of the user
    :return:
    """
    project_configuration_variables = load_configuration_variables(
        application_folder=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        experiment_name=configuration_file)

    # Feed Variables To Trainer Object
    trainer_object = Trainer(**project_configuration_variables)

    if train_mode:
        trainer_object.train()

    if evaluate_mode:
        trainer_object.evaluate(iteration=model_iteration)

    if predict_mode:
        prediction = trainer_object.predict(iteration=model_iteration, smile_string=smile_string)
        print(prediction)

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
    parser.add_argument("--index", type=int, default=0,
                        help='model index iteration that we would like to use')
    parser.add_argument("--smile", type=str, default='',
                        help='smile string that we would like to predict')

    parsed_arguments, un_parsed_arguments = parser.parse_known_args()

    # Make sure that arguments are coherent
    # At least one mode has to be True, either train, evaluate or predict and only one mode has to be True
    if not (parsed_arguments.train or parsed_arguments.evaluate or parsed_arguments.predict):
        print_red("Please Specify The Mode That You Would Like To Launch, either train, evaluate or predict")
        print_green("This is done by specifying either --train --evaluate or --predict")
        exit()
    if [parsed_arguments.train, parsed_arguments.evaluate, parsed_arguments.predict].count(True) > 1:
        print_red("Please Specify Only One Mode, either train, evaluate or predict")
        exit()

    # In case of evaluation or prediction make sure that index is different from 0
    if parsed_arguments.evaluate or parsed_arguments.predict:
        if not parsed_arguments.index:
            print_red("Please Specify The Iteration Of The Model You Would Like To Use")
            print_green("This is done by specifying --index=xxx")
            exit()

    # In case of prediction make sure that a smile string has been given
    if parsed_arguments.predict:
        if not parsed_arguments.smile:
            print_red("Please Specify A Smile String Of A Molecule That You Would Like To Predict The Properties")
            print_green("This is done by specifying --smile=xxx")
            exit()

    # Run the main function
    main(configuration_file=parsed_arguments.config,
         train_mode=parsed_arguments.train,
         evaluate_mode=parsed_arguments.evaluate,
         predict_mode=parsed_arguments.predict,
         model_iteration=parsed_arguments.index,
         smile_string=parsed_arguments.smile)
