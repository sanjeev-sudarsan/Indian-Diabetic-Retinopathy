import json
import gin
import logging
import os
import tensorflow as tf
import tensorflow_datasets as tfds


from input_pipeline.preprocessing import (
    augment,
    normalize_image,
    gc_preprocess,
    bt_preprocess,
)

from input_pipeline.tf_records import create_tfrecords


logger = logging.getLogger()


@gin.configurable
def load(run_paths, name, data_dir, binarize, preprocess_type):
    """
    Loads the specified dataset from tfrecords present in the directory in data dir.

    Parameters:
        run_paths (dict): Dictionary containing all the required paths.
        name (str): Name of the dataset.
        data_dir (str): Location of the dataset.
        binarize (Boolean): Specifies if the dataset should be binarized i.e. the number of classes is 2.
        preprocess_type (str): The preprocessing method which should be used.

    Returns:
        ds_train (tf.data.Dataset): The train dataset.
        ds_val (tf.data.Dataset): The validaion dataset.
        ds_test (tf.data.Dataset): The test dataset.
        ds_info (tfds.core.DatasetInfo): Tensorflow dataset info object.
    """

    if name == "idrid_dataset":
        logger.info(f"Preparing dataset {name}...")
        try:
            edit_ds_info(name, run_paths["path_datasets"], binarize)
            (ds_train, ds_val, ds_test), ds_info = tfds.load(
                name=name,
                split=["train", "val", "test"],
                with_info=True,
                data_dir=run_paths["path_datasets"],
                as_supervised=True,
            )

        except:
            logger.info(f"creating tfrecords for the dataset {name}...")

            # creates the TFRecords for the idird dataset if it is not available.
            create_tfrecords(data_dir, "IDRID_dataset", run_paths)
            edit_ds_info(name, run_paths["path_datasets"], binarize)
            (ds_train, ds_val, ds_test), ds_info = tfds.load(
                name=name,
                split=["train", "val", "test"],
                with_info=True,
                data_dir=run_paths["path_datasets"],
                as_supervised=True,
            )
        # Binarize the dataset to two classes.
        if binarize:
            logger.info(
                f"Reducing the number of classes to two for the dataset {name}..."
            )
            ds_train = ds_train.map(
                lambda image, label: (image, 0)
                if tf.math.less_equal(label, 1)
                else (image, 1)
            )
            ds_val = ds_val.map(
                lambda image, label: (image, 0)
                if tf.math.less_equal(label, 1)
                else (image, 1)
            )

            ds_test = ds_test.map(
                lambda image, label: (image, 0)
                if tf.math.less_equal(label, 1)
                else (image, 1)
            )

        if preprocess_type == "bt_preprocess":
            preprocess_method = bt_preprocess
        elif preprocess_type == "gc_preprocess":
            preprocess_method = gc_preprocess
        else:
            raise ValueError

        # Update ds_info so that label and image data match the dataset.
        # This is done because preprocessing changes the image shape and binarize changes the label data.
        logger.info(
            f"applying the preprocessing method {preprocess_type} to the dataset {name}..."
        )

        # Apply preprocessing.
        ds_train = ds_train.map(
            lambda image, label: (
                tf.numpy_function(preprocess_method, [image], tf.uint8),
                label,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        ds_val = ds_val.map(
            lambda image, label: (
                tf.numpy_function(preprocess_method, [image], tf.uint8),
                label,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        ds_test = ds_test.map(
            lambda image, label: (
                tf.numpy_function(preprocess_method, [image], tf.uint8),
                label,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "eyepacs":
        logger.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            "diabetic_retinopathy_detection/btgraham-300",
            split=["train", "validation", "test"],
            shuffle_files=True,
            with_info=True,
            data_dir=data_dir,
        )

        def _preprocess(img_label_dict):
            return img_label_dict["image"], img_label_dict["label"]

        ds_train = ds_train.map(
            _preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        ds_val = ds_val.map(
            _preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        ds_test = ds_test.map(
            _preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        return prepare(ds_train, ds_val, ds_test, ds_info)

    elif name == "mnist":
        logger.info(f"Preparing dataset {name}...")
        (ds_train, ds_val, ds_test), ds_info = tfds.load(
            "mnist",
            split=["train[:90%]", "train[90%:]", "test"],
            shuffle_files=True,
            as_supervised=True,
            with_info=True,
            data_dir=data_dir,
        )

        return prepare(ds_train, ds_val, ds_test, ds_info, 16, False)

    else:
        raise ValueError


@gin.configurable
def prepare(ds_train, ds_val, ds_test, ds_info, batch_size, caching):
    # Prepare the train dataset
    logger.info("preparing the dataset")

    if caching:
        ds_train = ds_train.cache()

    # Augment the train dataset.
    ds_train = ds_train.map(
        augment,
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    ds_train = ds_train.map(
        normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples // 10)
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.repeat(-1)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare validation dataset
    ds_val = ds_val.batch(batch_size)
    if caching:
        ds_val = ds_val.cache()
    ds_val = ds_val.map(
        normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    # Prepare test dataset
    ds_test = ds_test.batch(batch_size)
    if caching:
        ds_test = ds_test.cache()
    ds_test = ds_test.map(
        normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info


def edit_ds_info(name, data_dir, binarize):
    """
    Updates the ds_info so that the label information and image shape match the dataset.

    Parameters:
        name (str): Name of the dataset.
        data_dir (str): Location of the dataset.
        binarize (Boolean): Specifies if the dataset should be binarized i.e. the number of classes is 2.
    """

    if binarize:
        with open(os.path.join(data_dir, name, "1.0.0", "features.json"), "r") as f:
            data = json.load(f)

            # Set the number of classes to 2.
            data["content"]["label"]["content"]["num_classes"] = 2

            # Set the shape of the image to (300,300,3).
            data["content"]["image"]["content"]["shape"] = [300, 300, 3]
        with open(os.path.join(data_dir, name, "1.0.0", "features.json"), "w") as f:
            json.dump(data, f, indent=4)

        # Set the updated labels.
        with open(os.path.join(data_dir, name, "1.0.0", "label.labels.txt"), "w") as f:
            f.write("0\n1")
    else:
        with open(os.path.join(data_dir, name, "1.0.0", "features.json"), "r") as f:
            data = json.load(f)

            # Set the number of classes to 5.
            data["content"]["label"]["content"]["num_classes"] = 5

            # Set the shape of the image to (300,300,3).
            data["content"]["image"]["content"]["shape"] = [300, 300, 3]
        with open(os.path.join(data_dir, name, "1.0.0", "features.json"), "w") as f:
            json.dump(data, f, indent=4)

        # Set the updated labels.
        with open(os.path.join(data_dir, name, "1.0.0", "label.labels.txt"), "w") as f:
            f.write("0\n1\n2\n3\n4")
