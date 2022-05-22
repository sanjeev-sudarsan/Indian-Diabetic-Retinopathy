"""
Definitions for available model architectures
"""
import gin
import os
import tensorflow as tf

from models.layers import vgg_block


@gin.configurable
def vgg_like(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """
    Defines a VGG-like architecture.

    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate

    Returns:
        (keras.Model): keras model object
    """

    assert n_blocks > 0, "Number of blocks has to be at least 1."

    inputs = tf.keras.Input(input_shape)
    out = vgg_block(inputs, base_filters)
    for i in range(2, n_blocks):
        out = vgg_block(out, base_filters * 2 ** (i))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(n_classes)(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="vgg_like")


@gin.configurable
class TransferLearningModels(tf.keras.Model):
    """
    Defines transfer learning models

    Parameters:
        model_type (string): name of the transfer learning model to use
        input_shape (tuple: 3): input shape of the neural network
        number_of_classes (int): number of classes in the dataset
        first_fine_tune_layer (int): layer from which to start fine tuning the model

    Returns:
        (keras.Model): keras model object
    """

    model_types = ["resnet50", "efficientnet"]

    def __init__(
        self,
        model_type,
        input_shape,
        number_of_classes,
        dropout_rate,
        reg_factor,
        first_fine_tune_layer=None,
        **kwargs
    ):

        assert number_of_classes > 0

        self.dropout_rate = dropout_rate

        self.reg_factor = reg_factor

        super(TransferLearningModels, self).__init__()

        input_tensor = tf.keras.Input(shape=input_shape)

        if model_type == "resnet50":
            self.base_model = tf.keras.applications.resnet50.ResNet50(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet",
                input_tensor=input_tensor,
            )
            if first_fine_tune_layer is None:
                first_fine_tune_layer = 160

        elif model_type == "efficientnet":
            self.base_model = tf.keras.applications.efficientnet.EfficientNetB4(
                input_shape=input_shape,
                include_top=False,
                weights="imagenet",
                input_tensor=input_tensor,
            )
            if first_fine_tune_layer is None:
                first_fine_tune_layer = 456

        self.base_model.trainable = False
        self.first_fine_tune_layer = first_fine_tune_layer

        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_rate)
        self.classification_layer = tf.keras.layers.Dense(
            units=number_of_classes, activation="softmax"
        )

    def unfreeze_top_layers(self):
        """
        The ResNet50 model       -> 175 layers, excluding top
        The EfficientNetB4 model -> 474 layers, excluding top
        """

        for layer in self.base_model.layers[self.first_fine_tune_layer :]:
            layer.trainable = True

    def call(self, inputs, training=False):
        """
        Tensorflow tutorial:
        When you set layer.trainable = False, the BatchNormalization layer will
        run in inference mode,
        and will not update its mean and variance statistics.
        When you unfreeze a model that contains BatchNormalization layers
        in order to do fine-tuning, you should keep the BatchNormalization layers in
        inference mode by passing training = False when calling the base model.
        Otherwise, the updates applied to the non-trainable weights will destroy
        what the model has learned.
        """
        x = self.base_model(inputs=inputs, training=False)
        x = self.global_average_layer(x)
        if training:
            x = self.dropout_layer(x)
        return self.classification_layer(x)

    def expand_base_model_and_build_functional_model(self):
        base_model_conv_layer = None

        for layer in reversed(self.base_model.layers):
            if len(layer.output_shape) == 4:
                base_model_conv_layer = layer.name
                break

        x_2 = self.global_average_layer(
            self.base_model.get_layer(base_model_conv_layer).output
        )
        out = self.classification_layer(x_2)
        return tf.keras.Model(inputs=self.base_model.input, outputs=out)

    def add_regularization(self, run_paths):
        """
        Enables regularization in transfer learning models.

        Parameters:
            run_paths (dict): Dictionary containing all the required paths.
            regularizer (tf.keras.regulrizers object): The regularizer for the layers.

        Returns:
            A tf.keras.model object

        """
        regularizer = tf.keras.regularizers.l2(self.reg_factor)
        if not isinstance(regularizer, tf.keras.regularizers.Regularizer):
            print("Regularizer must be a subclass of tf.keras.regularizers.Regularizer")
            return self.base_model

        for layer in self.base_model.layers:
            for attr in ["kernel_regularizer"]:
                if hasattr(layer, attr):
                    setattr(layer, attr, regularizer)

        # When we change the layers attributes, the change only happens in the model config file
        model_json = self.base_model.to_json()

        # Save the weights before reloading the model.
        tmp_weights_path = os.path.join(run_paths["path_weights"], "tmp_weights.h5")
        self.base_model.save_weights(tmp_weights_path)

        # load the model from the config
        self.base_model = tf.keras.models.model_from_json(model_json)

        # Reload the model weights
        self.base_model.load_weights(tmp_weights_path, by_name=False)

        base_model_conv_layer = None

        for layer in reversed(self.base_model.layers):
            if len(layer.output_shape) == 4:
                base_model_conv_layer = layer.name
                break

        x_2 = self.global_average_layer(
            self.base_model.get_layer(base_model_conv_layer).output
        )
        out = self.classification_layer(x_2)
        return tf.keras.Model(inputs=self.base_model.input, outputs=out)
