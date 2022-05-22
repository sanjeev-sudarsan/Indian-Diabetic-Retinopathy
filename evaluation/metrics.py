import tensorflow as tf
import pandas as pd
import seaborn as sn
import io
import numpy as np
import matplotlib.pyplot as plt


def plot_to_image(figure):
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


class ConfusionMatrix(tf.keras.metrics.Metric):
    """
    This is a class which computes the confusion matrix.

    Attributes:
        num_classes (int): The number of classes of the input.
        name (str): The Name of the metric.
    """

    def __init__(self, num_classes, name="confusion_matrix", **kwargs):
        """
        The constructor for ConfusionMatrix class.

        Parameters:
            num_classes (int): The number of classes of the input.
            name (str): The Name of the metric.
        """

        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.conf_mat = self.add_weight(
            "total", shape=(num_classes, num_classes), initializer="zeros"
        )

    def reset_states(self):
        """
        Resets the state of the confusion matrix.
        """

        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, labels, predictions):
        """
        Updates the state of the confusion matrix.

        Parameters:
            labels (tf.tensor): The expected labels.
            Predictions (tf.tensor): The predicted labels.

        Returns:
            The updated confusion matrix.
        """

        self.conf_mat.assign_add(self.confusion_matrix(labels, predictions))
        return self.conf_mat

    def result(self):
        """
        Returns the confusion matrix as a numpy array.
        """

        return self.conf_mat.numpy().astype(np.int32)

    def confusion_matrix(self, labels, predictions):
        """
        create a confusion matrix for the given labels and predictions.

        Parameters:
            labels (tf.tensor): The expected labels.
            Predictions (tf.tensor): The predicted labels.

        Returns:
            The created confusion matrix.
        """

        predictions_decoded = tf.argmax(predictions, axis=1)
        cm = tf.math.confusion_matrix(
            labels, predictions_decoded, dtype=tf.float32, num_classes=self.num_classes
        )
        return cm

    def plot_confusion_matrix(self):
        """
        Plots the confusion matrix as a heatmap and returns the image.
        """

        matrix = self.conf_mat.numpy().astype(np.int32)
        df_cm = pd.DataFrame(
            matrix,
            index=["NRDR", "RDR"],
            columns=["NRDR", "RDR"],
        )
        plt.figure(figsize=(12, 12))
        sn_plot = sn.heatmap(
            df_cm,
            annot=True,
            fmt="d",
            cbar=False,
            annot_kws={"size": 100},
            cmap="Blues",
        )
        plt.xlabel("Predictions", fontsize=40)
        plt.ylabel("Labels", fontsize=40)
        sn_plot.xaxis.set_tick_params(labelsize=40)
        sn_plot.yaxis.set_tick_params(labelsize=40)
        figure = sn_plot.get_figure()
        image = plot_to_image(figure)
        return image
