'''
Script that contains the Flask application for property prediction
'''
from flask import Flask, request
from utils import parse_arguments_and_get_trainer_parameters
from trainer.trainer import Trainer

# Defining the Flask application
app = Flask(__name__)

# Get The Trainer Model Based On User Input
project_configuration_variables, model_iteration = parse_arguments_and_get_trainer_parameters()
trainer_object = Trainer(**project_configuration_variables)
trainer_object.load_and_restore_model(iteration=model_iteration)


@app.route('/predict')
def main():
    """Function that is going to launch the prediction"""
    smile_string = request.args.get("smile_string")

    prediction = trainer_object.predict_on_single_smile_string(smile_string)

    return prediction


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)
