import gin
import io
import os
import logging
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from evaluation.grad_cam import GradCAM, overlay_gradCAM
from evaluation.guided_backprop import GuidedBackprop, deprocess_image

logger = logging.getLogger()


@gin.configurable
class Visualizer(object):
    """
    This is a class which performs deep visualisation.

    Attributes:
        model (keras.model): Keras model object.
        run_paths (dict): Dictionary containing all the required paths.
        ds_test (tf.dataset): Tensorflow test dataset.
        ds_info (tfds.core.DatasetInfo): Tensorflow dataset info object.
        run_paths (dict): Dictionary containing all the required paths.
        layer_name (str): The name of the layer for which visualization has to be performed.
        checkpoint_name (str): Name of the checkpoint to be loaded for visualization.
    """

    def __init__(self, model, run_paths, ds_test, ds_info, layer_name, checkpoint_name):
        """
        The constructor for Visualizer class.

        Parameters:
            model (keras.model): Keras model object.
            run_paths (dict): Dictionary containing all the required paths.
            ds_test (tf.dataset): Tensorflow test dataset.
            ds_info (tfds.core.DatasetInfo): Tensorflow dataset info object.
            run_paths (dict): Dictionary containing all the required paths.
            layer_name (str): The name of the layer for which visualization has to be performed.
            checkpoint_name (str): Name of the checkpoint to be loaded for visualization.
        """
        self.model = model
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        self.ds_test = ds_test
        self.ckpt_dir = run_paths["path_ckpts"]
        self.len = ds_info.splits["test"].num_examples

        # restore checkpoint
        self.manager = tf.train.CheckpointManager(
            checkpoint=self.checkpoint, max_to_keep=3, directory=self.ckpt_dir
        )
        if checkpoint_name == None:
            self.checkpoint.restore(self.manager.latest_checkpoint).expect_partial()
        else:
            checkpoint_name = os.path.join(run_paths["path_ckpts"], checkpoint_name)
            try:
                self.checkpoint.restore(checkpoint_name).expect_partial()
            except:
                logger.info("Invalid checkpoint name. Loading the latest checkpoint.")
                self.checkpoint.restore(self.manager.latest_checkpoint).expect_partial()

        self.checkpoint.restore(self.manager.latest_checkpoint).expect_partial()

        # self.model = self.model.expand_base_model_and_build_functional_model()

        # create gradcam object
        self.layer_name = layer_name
        self.gradcam = GradCAM(self.model, self.layer_name)

        # create guided backpropagation object
        self.gb = GuidedBackprop(self.model, self.layer_name)

        # tensorboard summary writers
        tb_vis_dir = run_paths["path_vis_summary"]
        self.vis_writer = tf.summary.create_file_writer(tb_vis_dir)

    def visualize(self):
        logger.info("Started visualization for model {}".format(self.model.name))
        with tqdm(total=self.len, desc="Visualization") as pbar:
            for idx, (image, label) in enumerate(self.ds_test.unbatch()):
                step = idx + 1
                image_expanded_dim = tf.expand_dims(image, axis=0)

                # compute Grad-CAM
                given_class_gradcam, pred_class_gradcam = self.gradcam.compute_heatmap(
                    image_expanded_dim, label
                )

                # Compute guided backpropagation gradients
                gb_image = self.gb.guided_backprop(image_expanded_dim)

                # Process the Grad-cam and gradients to get heatmap overlayed image and Guided Grad-CAM image.
                fig, axs = plt.subplots(
                    2, 2, sharex=True, sharey=True, constrained_layout=True
                )
                axs[0][0].set_title("Preprocessed image", fontsize=20)
                axs[0][0].set_xticks([])
                axs[0][0].set_yticks([])
                axs[0][0].imshow(image)
                axs[0][1].set_title("Grad-CAM", fontsize=20)
                axs[0][1].set_xticks([])
                axs[0][1].set_yticks([])
                axs[0][1].imshow(
                    cv2.cvtColor(
                        overlay_gradCAM(image, pred_class_gradcam), cv2.COLOR_BGR2RGB
                    )
                )
                axs[1][0].set_title("Guided Backpropagation", fontsize=20)
                axs[1][0].set_xticks([])
                axs[1][0].set_yticks([])
                axs[1][0].imshow(
                    cv2.cvtColor(
                        deprocess_image(gb_image),
                        cv2.COLOR_BGR2RGB,
                    )
                )
                axs[1][1].set_title("Guided Grad-CAM", fontsize=20)
                axs[1][1].set_xticks([])
                axs[1][1].set_yticks([])
                axs[1][1].imshow(
                    cv2.cvtColor(
                        deprocess_image(gb_image * pred_class_gradcam),
                        cv2.COLOR_BGR2RGB,
                    )
                )
                final_image = self._plot_to_image(fig)

                # Write image to tensorboard.
                with self.vis_writer.as_default():
                    tf.summary.image("Deep visualization", final_image, step=step)
                pbar.update(1)
        logger.info("Visualization completed for model {}".format(self.model.name))

    def _plot_to_image(self, figure):
        """
        Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call.

        Parameters:
            figure (matplotlib.figure.Figure): A matplotlib plot.

        Returns:
            image (tf.tensor): A tensor which can be saved as an image.
        """

        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format="png")

        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)

        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)

        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image
