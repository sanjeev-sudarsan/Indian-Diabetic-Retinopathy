import gin
import numpy as np
import cv2
import imutils
import random
import tensorflow as tf


def crop(image):
    """
    Crops the rows and columns in which all the values are zero.

    Parameters:
        image (numpy.ndarray): The image to be cropped.

    Returns.
        image (numpy.ndarray): The cropped image.

    """

    # If the image is 2-dimensional.
    if image.ndim == 2:
        y_nonzero, x_nonzero = np.nonzero(image)
        return image[
            np.min(y_nonzero) : np.max(y_nonzero), np.min(x_nonzero) : np.max(x_nonzero)
        ]

    # If the image is 3-dimensional.
    elif image.ndim == 3:
        z_nonzero, y_nonzero, _ = np.nonzero(image)
        return image[
            np.min(z_nonzero) : np.max(z_nonzero),
            np.min(y_nonzero) : np.max(y_nonzero),
            :,
        ]


def gc_preprocess(image):
    """
    Extracts the green channel from the image, highlights the exudates.

    Parameters:
        image (numpy.ndarray): The image to be preprocessed.

    Returns.
        image (numpy.ndarray): The preprocessed image.
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Grayscale conversion.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Get binary image which can be used to mask preprocessing artifacts.
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    # Get the green channel
    _, g, _ = cv2.split(image)

    # Apply CLAHE histogram equalization.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    image = clahe.apply(g)

    # Use the binary image to mask artifacts.
    image = cv2.bitwise_and(image, thresh)

    # Get the highest value of the pixels in the image.
    # The optic disk usually has the highest value.
    a = np.amax(image)
    b = a - 5
    c = a - 70

    # Get a binary image which masks all pixels except the ones in the optic disk.
    _, thresh2 = cv2.threshold(image, b, 20, cv2.THRESH_BINARY)

    # Get a binary image which masks all pixels except the ones in the exudates and optic disk.
    _, thresh3 = cv2.threshold(image, c, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (100, 100))
    thresh2 = cv2.dilate(thresh2, kernel, iterations=4)

    # Remove the pixels of the optic disk from the binary image.
    # This ensures that only the exudates are highlighted and the optic disk is not highlighted.
    thresh4 = cv2.bitwise_xor(thresh2, thresh3, mask=None)

    # Add the binary image to the green channel effectively highlighting the exudates.
    image = image + thresh4

    # Use the binary image to mask artifacts.
    image = cv2.bitwise_and(image, thresh)

    # Merge the 2D image to form a 3D image.
    image = cv2.merge((image, image, image))

    image = crop(image)
    image = resize(image)
    return image


def bt_preprocess(image):
    """
    Applies Graham preprocessing to the image

    Parameters:
        image (numpy.ndarray): The image to be preprocessed.

    Returns.
        image (numpy.ndarray): The preprocessed image.
    """

    # Grayscale conversion.
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Subtract the local average.
    image = cv2.addWeighted(
        image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX=100), -4, 128
    )

    # Get binary image which can be used to mask preprocessing artifacts.
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    # Merge the 2D binary mask to form a 3D binary mask.
    thresh_3d = cv2.merge((thresh, thresh, thresh))

    # Use the binary image to mask artifacts
    image = cv2.bitwise_and(image, thresh_3d)

    image = crop(image)
    image = resize(image)
    return image


@gin.configurable
def resize(image, desired_size):
    """
    Resizes the image to the desired size and maintain aspect ratio.
    The dimension of the image is desired_size x desired_size

    Parameters:
        image (numpy.ndarray): The image to be resized.
        desired_size (int): The size to which the image should be resized.

    Returns:
        image (numpy.ndarray): The resized image.
    """

    # Resize image and maintain aspect ratio.
    image = imutils.resize(image, width=desired_size)
    height = image.shape[0]

    # Get the size of the padding.
    if (desired_size - height) % 2 == 0:
        top = int((desired_size - height) / 2)
        bottom = top
    else:
        top = int((desired_size - height) / 2)
        bottom = top + 1

    # Pad the image
    image = cv2.copyMakeBorder(
        image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, None, value=0
    )

    if image.ndim == 2:
        image = np.reshape(image, (desired_size, desired_size, 1))
    return image


def normalize_image(image, label):
    """
    Normalizes the image so that values are between 0 and 1.

    Parameters:
        image (tf.tensor) = The image to be normalized.
        label (tf.tensor) = The label of the image to be normalized.

    Returns:
        image (tf.tensor) = The normalized image.
        label (tf.tensor) = The label of the normalized image.
    """

    tf.cast(image, tf.float32) / 255.0
    # image = tf.image.per_image_standardization(image)

    return image, label


@gin.configurable
def augment(image, label):
    """
    Applies random transformations to the image.

    Parameters:
        image (tf.tensor) = The image to be transformed.
        label (tf.tensor) = The label of the image to be transformed.

    Returns:
        image (tf.tensor) = The transformed image.
        label (tf.tensor) = The label of the transformed image.
    """

    flip_options = [
        tf.image.flip_left_right,
        tf.image.flip_up_down,
        tf.image.random_flip_left_right,
        tf.image.random_flip_up_down,
        tf.image.rot90,
    ]
    flip = random.choice(flip_options)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.1, upper=0.5)
    image = tf.image.random_hue(image, max_delta=0.4)
    image = tf.image.random_saturation(image, lower=5, upper=10)
    image = flip(image)
    return image, label
