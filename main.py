import logging
import gin
from absl import app, flags
import pathlib
import tensorflow as tf
from train import Trainer
from evaluation.eval import Evaluator
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like, TransferLearningModels
from evaluation.visualization import Visualizer
import wandb

FLAGS = flags.FLAGS
flags.DEFINE_boolean("train", False, "Specify whether to train or evaluate a model.")
flags.DEFINE_boolean("eval", False, "Specify whether to evaluate a model.")
flags.DEFINE_boolean("visualize", False, "Specify whether to perfrom Deep Visualization.")
flags.DEFINE_string(
    name="dir_name", default="dr_run", help="Specify the name of the run folder."
)

tf.random.set_seed(245)


@gin.configurable
def get_model(ds_info, types):
    """
    Returns the specified model.

    Parameters:
        ds_info (tfds.core.DatasetInfo): Tensorflow dataset info object.
        types (str): The name of the model to be used.
    Returns:
        model (keras.model): Keras model object.
    """

    n_classes = ds_info.features["label"].num_classes
    input_shape = ds_info.features["image"].shape
    if types == "vgg_like":
        model = vgg_like(input_shape=input_shape, n_classes=n_classes)
    elif types == "resnet50":
        model = TransferLearningModels(
            model_type=types, input_shape=input_shape, number_of_classes=n_classes
        )
    elif types == "efficientnet":
        model = TransferLearningModels(
            model_type=types, input_shape=input_shape, number_of_classes=n_classes
        )
    else:
        raise ValueError("Invalid model")
    return model


def main(argv):
    """
    The main function. It creates the run paths, sets the config and trains and evaluates the model.
    """
    
    # generate folder structures
    run_paths = utils_params.gen_run_folder(FLAGS.dir_name)
    # gin-config
    utils_params.save_config(run_paths["file_gin"], gin.config_str())
    utils_misc.set_loggers(paths=run_paths, logging_level=logging.INFO)

    wandb.init(project=FLAGS.dir_name)
    # setup pipeline
    ds_train, ds_val, ds_test, ds_info = datasets.load(run_paths)

    # model
    model = get_model(ds_info=ds_info)
    model = model.add_regularization(run_paths)

    if FLAGS.train:
        trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths)
        for _ in trainer.train():
            continue
    if FLAGS.eval:
        evaluator = Evaluator(model, ds_test, ds_info, run_paths)
        evaluator.evaluate_model()
    if FLAGS.visualize:
        visualizer = Visualizer(model, run_paths, ds_test, ds_info)
        visualizer.visualize()


if __name__ == "__main__":
    gin_config_path = pathlib.Path(__file__).parent / "configs" / "config.gin"
    gin.parse_config_files_and_bindings([gin_config_path], [])
    app.run(main)
