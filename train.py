"""
Contains Trainer class with it's helper function

Authors: Simhadri and Sanjeev
"""
import tensorflow as tf
import gin
import logging
from evaluation.metrics import ConfusionMatrix
import wandb


training_logger = logging.getLogger("training")


@gin.configurable
class Trainer:
    """
    trainer class definition
    """

    def __init__(
        self,
        model,
        ds_train,
        ds_val,
        ds_info,
        run_paths,
        learning_rate,
        log_interval,
        ckpt_interval,
        total_steps,
    ):
        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_info = ds_info
        self.run_paths = run_paths
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.total_steps = total_steps
        self.num_classes = ds_info.features["label"].num_classes

        # Loss function and Optimizer
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name="train_loss")
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="train_accuracy"
        )
        self.train_confusion_matrix = ConfusionMatrix(self.num_classes)

        self.val_loss = tf.keras.metrics.Mean(name="val_loss")
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name="val_accuracy"
        )

        # tesnsorboard summary writers
        tb_train_dir = run_paths["path_train_summary"]
        tb_val_dir = run_paths["path_val_summary"]
        self.train_writer = tf.summary.create_file_writer(tb_train_dir)
        self.val_writer = tf.summary.create_file_writer(tb_val_dir)

        # Checkpoint manager
        ckpt_dir = run_paths["path_ckpts"]
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, model=self.model
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory=ckpt_dir,
            max_to_keep=4,
            checkpoint_name=self.model.name,
        )

    @tf.function
    def train_step(self, images, labels):
        """
        train a single step for given images and labels
        """
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_confusion_matrix.update_state(labels, predictions)
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def val_step(self, images, labels):
        """
        evaluate a single step for given images and labels
        """
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        v_loss = self.loss_object(labels, predictions)

        self.val_loss(v_loss)
        self.val_accuracy(labels, predictions)

    def train(self):
        """
        main training loop with tensorboard summary and checkpoint saving
        """
        # with tf.profiler.experimental.Profile(self.run_paths["path_train_summary"]):
        for idx, (images, labels) in enumerate(self.ds_train):
            step = idx + 1
            self.train_step(images, labels)

            if step % self.log_interval == 0:
                # Reset validation metrics
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()

                for val_images, val_labels in self.ds_val:
                    self.val_step(val_images, val_labels)

                template = "Step {}, Train Loss: {}, Train Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}"
                training_logger.info(
                    template.format(
                        idx + 1,
                        self.train_loss.result(),
                        self.train_accuracy.result() * 100,
                        self.val_loss.result(),
                        self.val_accuracy.result() * 100,
                    )
                )

                # Write summary to tensorboard
                # add summary for other types of tensorboard
                with self.train_writer.as_default():
                    tf.summary.scalar("Loss", self.train_loss.result(), step=step)
                    tf.summary.scalar(
                        "Accuracy", self.train_accuracy.result(), step=step
                    )
                    tf.summary.image(
                        "Confusion matrix",
                        self.train_confusion_matrix.plot_confusion_matrix(),
                        step=step,
                    )
                wandb.log(
                    {
                        "train_loss": self.train_loss.result(),
                        "train_accuracy": self.train_accuracy.result(),
                        "validation_loss": self.val_loss.result(),
                        "val_accuracy": self.val_accuracy.result(),
                        "step": step,
                    }
                )
                with self.val_writer.as_default():
                    tf.summary.scalar("Loss", self.val_loss.result(), step=step)
                    tf.summary.scalar("Accuracy", self.val_accuracy.result(), step=step)

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.val_accuracy.result().numpy()

            # Save checkpoint
            if step % self.ckpt_interval == 0:
                # print(f'Saving checkpoint to {self.run_paths["path_ckpts"]}.')
                self.manager.save()

            # Save final checkpoint
            if step % self.total_steps == 0:
                self.manager.save()
                training_logger.info(f"Finished training after {step} steps.")
                return self.val_accuracy.result().numpy()

    def profiling(self):
        with tf.profiler.experimental.Profile(self.run_paths["path_train_summary"]):
            self.train()
