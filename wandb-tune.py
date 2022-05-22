import logging
import wandb
import pathlib
from train import Trainer
from input_pipeline import datasets
from utils import utils_params, utils_misc
from evaluation.visualization import Visualizer
from main import get_model
import gin

# os.environ['WANDB_MODE'] = 'offline'

# Name of the folder and wandb project for hyperparameter tuning.
dir_name = "run"

def train_func():
    """
    Trains the model using the hyperparameters set in config.
    """

    with wandb.init() as run:
        gin.clear_config()
        # Hyperparameters
        bindings = []
        for key, value in run.config.items():
            bindings.append(f"{key}={value}")

        # generate folder structures
        run_paths = utils_params.gen_run_folder(dir_name)

        # gin-config
        gin_config_path = pathlib.Path(__file__).parent / "configs" / "config.gin"
        gin.parse_config_files_and_bindings([gin_config_path], bindings)
        utils_params.save_config(run_paths["file_gin"], gin.config_str())

        # set loggers
        utils_misc.set_loggers(paths=run_paths, logging_level=logging.INFO)

        # setup pipeline
        ds_train, ds_val, ds_test, ds_info = datasets.load(run_paths)

        # model
        model = get_model(ds_info=ds_info)
        model = model.add_regularization(run_paths)

        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue
        # visualizer = Visualizer(model,run_paths,ds_test, ds_info)


# Configuration for the hyperparameter tuning.
sweep_config = {
    "program": "wandb-train.py",
    "command": ["python3", "wandb-train.py"],
    "name": dir_name,
    "parameters": {
        # "TransferLearningModels.dropout_rate": {"values": [0.2, 0.25, 0.3]},
        "TransferLearningModels.dropout_rate": {
            "distribution": "uniform",
            "min": 0.1,
            "max": 0.4,
        },
        "Trainer.learning_rate": {"values": [0.01, 0.02, 0.03, 0.001, 0.002, 0.003]},
        "TransferLearningModels.first_fine_tune_layer": {
            "distribution": "int_uniform",
            "min": 150,
            "max": 170,
            # "values": [451, 453, 455, 457, 459]
        },
        # "Trainer.total_steps": {"values": [2000, 3000]},
        "prepare.batch_size": {"values": [64, 128]},
        # "TransferLearningModels.reg_factor": {"values": [0.01, 0.001]},
        "TransferLearningModels.reg_factor": {
            "distribution": "log_uniform",
            "min": -6.9,
            "max": -4.6,
        },
    },
    "metric": {"goal": "maximize", "name": "val_accuracy"},
    "method": "random",
}

# Create a sweep 
sweep_id = wandb.sweep(sweep_config, project=dir_name)


# Perfrom the sweep
wandb.agent(sweep_id, function=train_func, count=25)
