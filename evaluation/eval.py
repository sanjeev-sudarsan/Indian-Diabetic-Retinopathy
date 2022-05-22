import gin
import logging
import os
from tqdm import tqdm
import tensorflow as tf
from evaluation.metrics import ConfusionMatrix


evaluation_logger = logging.getLogger("evaluation")


@gin.configurable
class Evaluator(object):
    """
    This is a class for performing evaluating the model.

    Attributes:
        model (keras.model): Keras model object.
        ds_test (tf.dataset): Tensorflow test dataset.
        ds_info (tfds.core.DatasetInfo): Tensorflow dataset info object.
        run_paths (dict): Dictionary containing all the required paths.
        checkpoint_name (str): Name of the checkpoint to be loaded for evaluation.

    """

    def __init__(self, model, ds_test, ds_info, run_paths, checkpoint_name):
        """
        The constructor for Evaluator class.

        Parameters:
            model (keras.model): Keras model object.
            ds_test (tf.dataset): Tensorflow test dataset.
            ds_info (tfds.core.DatasetInfo): Tensorflow dataset info object.
            run_paths (dict): Dictionary containing all the required paths.
            checkpoint_name (str): Name of the checkpoint to be loaded for evaluation.

        """

        self.model = model
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        self.ds_test = ds_test
        self.ds_info = ds_info
        self.num_classes = ds_info.features["label"].num_classes
        self.ckpt_dir = run_paths["path_ckpts"]
        self.manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint, max_to_keep=3, directory=self.ckpt_dir
        )

        # Metrics
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        )
        self.test_loss = tf.keras.metrics.Mean(name="test_loss")
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="test_accuracy"
        )
        self.test_confusion_matrix = ConfusionMatrix(self.num_classes)

        # Tesnsorboard summary writers
        tb_test_dir = run_paths["path_test_summary"]
        self.test_writer = tf.summary.create_file_writer(tb_test_dir)

        # Restore checkpoint
        if checkpoint_name == None:
            self.checkpoint.restore(self.manager.latest_checkpoint).expect_partial()
        else:
            checkpoint_name = os.path.join(run_paths["path_ckpts"], checkpoint_name)
            try:
                self.checkpoint.restore(checkpoint_name).expect_partial()
            except:
                evaluation_logger.info(
                    "Invalid checkpoint name. Loading the latest checkpoint."
                )
                self.checkpoint.restore(self.manager.latest_checkpoint).expect_partial()

    def evaluate_model(self):
        """
        Evaluates the model and saves the accuracy and confusion matrix in tensorboard.

        Returns:
            None.
        """

        evaluation_logger.info(
            "Started evaluation for model {}".format(self.model.name)
        )
        with tqdm(total=len(self.ds_test), desc="Evaluation") as pbar:
            for idx, (images, labels) in enumerate(self.ds_test):
                step = idx + 1

                # Test step
                self._test_step(images, labels)

                # Write loss, accuracy and confusion matrix to tensorboard.
                with self.test_writer.as_default():
                    tf.summary.scalar("Loss", self.test_loss.result(), step=step)
                    tf.summary.scalar(
                        "Accuracy", self.test_accuracy.result(), step=step
                    )
                    tf.summary.image(
                        "Confusion matrix",
                        self.test_confusion_matrix.plot_confusion_matrix(),
                        step=step,
                    )
                pbar.update(1)
        evaluation_logger.info(
            "Evaluation completed for model {}".format(self.model.name)
        )
        template = "Loss: {}, Accuracy: {}"
        evaluation_logger.info(
            template.format(
                self.test_loss.result(),
                self.test_accuracy.result() * 100,
            )
        )
        evaluation_logger.info("//////////////////////////////////////////////")

    def _test_step(self, images, labels):
        """
        Evaluates a single step for given images and labels.

        Paramters:
            images (tf.tensor): A batch of image tensors.
            labels (tf.tensor): A batch of corresponding label tensors.
        """
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        # Update confusion matrix
        self.test_confusion_matrix.update_state(labels, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
