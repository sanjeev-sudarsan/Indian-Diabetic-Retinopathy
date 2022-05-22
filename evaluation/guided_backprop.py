import gin
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K


@tf.custom_gradient
def guidedRelu(x):
    """
    A custom RELU activation.

    Parameters:
        x (tf.tensor): An input tensor.

    Returns:
        tf.nn.relu(x) (tf.tensor): A RELU activation of x.
        grad (tf.tensor): A custom gradient.

    """

    def grad(dy):
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy

    return tf.nn.relu(x), grad


@gin.configurable
class GuidedBackprop:
    """
    This is a class which computes Guided Backpropagation.

    Attributes:
        model (keras.model): Keras model object.
        layerName (str): The name of the layer for which Grad-CAM is computed.
        upsample_size (tuple): Size of the Grad-CAM heatmap.
    """

    def __init__(self, model, layerName, upsample_size):
        """
        The constructor for GuidedBackprop class.

        Parameters:
            model (keras.model): Keras model object.
            layerName (str): The name of the layer for which Grad-CAM is computed.
            upsample_size (tuple): Size of the Grad-CAM heatmap.
        """

        self.model = model
        self.layerName = layerName
        self.upsample_size = upsample_size
        if self.layerName == None:
            self.layerName = self._find_target_layer()
        self.gbModel = self._build_guided_model()

    def _find_target_layer(self):
        """
        finds and returns the name of the last CNN layer in the model.

        Returns:
            layer.name: The name of the last CNN layer in the model.
        """

        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply Guided Backpropagation")

    def _build_guided_model(self):
        """
        Builds and returns a model which can be used for guided backpropagation.

        Returns:
            gbModel (keras.model): A keras model which can be used for guided backpropagation.
        """

        gbModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output],
        )

        # Get all layers with activation.
        layer_dict = [
            layer for layer in gbModel.layers[1:] if hasattr(layer, "activation")
        ]

        # Replace standard RELU with custom RELU.
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu

        return gbModel

    def guided_backprop(self, images):
        """
        Performs guided backpropagation and returns the gradients.

        Parameters:
            Images (tf.tensor): An input tensor of the image file.

        Returns:
            saliency (numpy.ndarray): A numpy array containing the gradients computed using guided backpropagation.
        """

        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            outputs = self.gbModel(inputs)

        grads = tape.gradient(outputs, inputs)[0]

        # Resize gradient image to desired size
        saliency = cv2.resize(np.asarray(grads), self.upsample_size)

        return saliency


def deprocess_image(x):
    """
    Deprocesses the given image.

    Parameters:
        x (numpy.ndarray): The image to be deprocessed.

    Returns:
        x (numpy.ndarray): The deprocessed image.
    """

    # normalize image: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= x.std() + K.epsilon()
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == "channels_first":
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype("uint8")
    return x
