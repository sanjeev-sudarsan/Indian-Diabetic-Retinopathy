import gin
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import Model


@gin.configurable
class GradCAM:
    """
    This is a class which computes Grad-CAM.

    Attributes:
        model (keras.model): Keras model object.
        layerName (str): The name of the layer for which Grad-CAM is computed.
        upsample_size (tuple): Size of the Grad-CAM heatmap.
        const (float): A constant used to prevent division by zero.
    """

    def __init__(self, model, layerName, upsample_size, const):
        """
        The constructor for GradCAM class.

        Parameters:
            model (keras.model): Keras model object.
            layerName (str): The name of the layer for which Grad-CAM is computed.
            upsample_size (tuple): Size of the Grad-CAM heatmap.
            const (float): A constant used to prevent division by zero.
        """

        self.upsample_size = upsample_size
        self.const = const
        self.model = model
        self.layerName = layerName

        # Get the last CNN layer of the model if no layer is specified.
        if self.layerName == None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        """
        finds and returns the name of the last CNN layer in the model.

        Returns:
            layer.name (str): The name of the last CNN layer in the model.
        """

        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")

    def compute_heatmap(self, image, given_class_index):
        """
        Returns the Grad-CAM of the predicted and given classes.

        Parameters:
            image (tf.tensor): The input tensor.
            given_class_index (int): The ID of the class for which Grad-CAM heatmap should be computed.

        Returns:
            given_class_gradcam (numpy.ndarray): A numpy array containing the Grad-CAM for the given class.
            pred_class_gradcam (numpy.ndarray): A numpy array containing the Grad-CAM for the predicted class.
        """

        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output, self.model.output],
        )

        with tf.GradientTape(persistent=True) as tape:
            inputs = image
            (convOuts, preds) = gradModel(inputs)
            top_pred_index = tf.argmax(preds[0])
            given_class_loss = preds[:, given_class_index]
            pred_class_loss = preds[:, top_pred_index]

        # Gradients for the given class.
        given_class_grads = tape.gradient(given_class_loss, convOuts)

        # Gradients for the predicted class.
        pred_class_grads = tape.gradient(pred_class_loss, convOuts)

        # Discard batch dimension
        convOuts = convOuts[0]
        given_class_grads = given_class_grads[0]
        pred_class_grads = pred_class_grads[0]

        # Compute the Grad-CAM
        given_class_gradcam = self._get_cam(given_class_grads, convOuts)
        pred_class_gradcam = self._get_cam(pred_class_grads, convOuts)

        return given_class_gradcam, pred_class_gradcam

    def _get_cam(self, grads, convOuts):
        """
        Computes and returns the Grad-CAM from the gradients and output of the CNN.

        Parameters:
            grads (tf.tensor): The gradients of the loss with respect to the outputs of a CNN layer.
            convOuts (tf.tensor): Outputs of a CNN layer.

        Returns:
            cam3 (numpy.ndarray): A numpy array containing the computed Grad-CAM.
        """

        norm_grads = tf.divide(
            grads, tf.reduce_mean(tf.square(grads)) + tf.constant(self.const)
        )

        weights = tf.reduce_mean(norm_grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)

        cam = np.maximum(cam, 0)
        cam = cam / np.max(cam)
        cam = cv2.resize(cam, self.upsample_size, cv2.INTER_LINEAR)

        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1, 1, 3])
        return cam3


def overlay_gradCAM(img, cam3):
    """
    Overlays the Grad-CAM heatmap on the corresponding image.

    Parameters:
        img (tf.tensor): An image tensor.
        cam3 (numpy.ndarray): A numpy array containing the Grad-CAM of the respective image.

    Returns:
        A numpy array containing the computed Grad-CAM overlayed on the respective image.
    """
    img = img.numpy()
    cam3 = np.uint8(255 * cam3)

    # Get the heatmap from the Grad-CAM
    cam3 = cv2.applyColorMap(cam3, cv2.COLORMAP_JET)

    new_img = 0.3 * cam3 + 0.5 * img

    return (new_img * 255.0 / new_img.max()).astype("uint8")
