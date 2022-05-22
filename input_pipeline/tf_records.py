import glob
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
import tensorflow_datasets as tfds


ros = RandomOverSampler(random_state=245)


def add_to_tfds(run_paths, train_shard_length, val_shard_length, test_shard_length):
    """
    Adds metadata info so that the tfrecords can be loaded via tfds.

    Parameters:
        run_paths (dict): Dictionary containing all the required paths.
        train_shard_length (int): Number of image-label pairs stored in idrid_dataset-train.tfrecord files.
        val_shard_length (int): Number of image-label pairs stored in idrid_dataset-val.tfrecord files.
        test_shard_length (int): Number of image-label pairs stored in idrid_dataset-test.tfrecord files.
    """
    features = tfds.features.FeaturesDict(
        {
            "image": tfds.features.Image(shape=(2848, 4288, 3)),
            "label": tfds.features.ClassLabel(names=["0", "1", "2", "3", "4"]),
        }
    )

    split_infos = [
        tfds.core.SplitInfo(
            name="train",
            shard_lengths=train_shard_length,
            num_bytes=0,
        ),
        tfds.core.SplitInfo(
            name="val",
            shard_lengths=val_shard_length,
            num_bytes=0,
        ),
        tfds.core.SplitInfo(
            name="test",
            shard_lengths=test_shard_length,
            num_bytes=0,
        ),
    ]
    tfds.folder_dataset.write_metadata(
        data_dir=run_paths["path_idrid"],
        features=features,
        split_infos=split_infos,
        homepage="http://my-project.org",
        supervised_keys=("image", "label"),
        citation="""BibTex citation.""",
    )


def _bytes_feature(value):
    """
    Returns a bytes_list from a string / byte.
    """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """
    Returns an int64_list from a bool / enum / int / uint.
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def get_files(name, data_dir):
    """
    Returns lists of file paths of the images and numpy arrays containing the respective labels.

    Parameters:
        name (str): Name of the dataset.
        data_dir (str): Location of the dataset.

    Returns:
        train_image_files_ros (list): The list of file paths of the train images.
        train_labels_ros (numpy.ndarray): The numpy array containing the corresponding labels of the train images.
        val_image_files (list): The list of file paths of the validation images.
        val_labels (numpy.ndarray): The numpy array containing the corresponding labels of the validation images.
        test_image_files (list): The list of file paths of the test images.
        test_labels (numpy.ndarray): The numpy array containing the corresponding labels of the test images.
    """

    # Get the list of image file paths and label numpy arrays.
    train_image_files = glob.glob(os.path.join(data_dir, name, "images/train/*.jpg"))
    train_image_files = sorted(train_image_files, reverse=False)
    train_df = pd.read_csv(os.path.join(data_dir, name, "labels/train.csv"))
    train_df = train_df.sort_values(by=["Image name"], ascending=True)
    train_labels = train_df["Retinopathy grade"].to_numpy()
    test_image_files = glob.glob(os.path.join(data_dir, name, "images/test/*.jpg"))
    test_image_files = sorted(test_image_files, reverse=False)
    test_df = pd.read_csv(os.path.join(data_dir, name, "labels/test.csv"))
    test_df = test_df.sort_values(by=["Image name"], ascending=True)
    test_labels = test_df["Retinopathy grade"].to_numpy()

    # Allocate 25% of the train images for validation.
    val_length = int(len(train_image_files) * 0.25)
    val_image_files = train_image_files[:val_length]
    val_labels = train_labels[:val_length]
    train_image_files = train_image_files[val_length:]
    train_labels = train_labels[val_length:]

    # Oversample the train data to balance the dataset.
    train_image_files = np.array(train_image_files)
    train_image_files = np.reshape(train_image_files, (-1, 1))
    train_image_files_ros, train_labels_ros = ros.fit_resample(
        train_image_files, train_labels
    )
    train_image_files_ros = np.reshape(train_image_files_ros, (-1)).tolist()

    return (
        train_image_files_ros,
        train_labels_ros,
        val_image_files,
        val_labels,
        test_image_files,
        test_labels,
    )


def image_example(image_file, label):
    """
    Creates a tf.train.Example file for each image, label pair

    Parameters:
        image_file (str): Path of the image.
        label (int): Corresponding label of the image.

    Returns:
        A tf.train.Example file
    """
    image_string = open(image_file, "rb").read()
    feature = {"image": _bytes_feature(image_string), "label": _int64_feature(label)}
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecords(image_files, labels, run_paths, type="train", size=100):
    """
    Writes the example files as tfrecords in data_dir.

    Parameters:
        image_files (list): List of image file paths.
        labels (numpy.ndarray): Numpy array containing the respective labels of the images
        run_paths (dict): Dictionary containing all the required paths.
        type (str): The type of the data i.e. train, validation or test.
        size (int): Maximum number of image-label pairs stored in each TFRecord file.

    Returns:
        f_len (int): Number of image-label pairs stored in the TFRecords.
    """
    n_files = int(len(image_files) / size) + int(len(image_files) % size != 0)
    f_len = []
    for f in range(n_files):
        n_images = min(size, len(image_files) - f * size)
        f_len.append(n_images)
        filename = os.path.join(
            run_paths["path_idrid"],
            "idrid_dataset-" + type + ".tfrecord-0000{}-of-0000{}".format(f, n_files),
        )
        with tf.io.TFRecordWriter(filename) as writer:
            for i in range(n_images):
                image = image_files[size * f + i]
                label = labels[size * f + i]
                tf_example = image_example(image, label)
                writer.write(tf_example.SerializeToString())
    return f_len


def create_tfrecords(data_dir, name, run_paths):
    """
    Creates TFrecord files for the IDRID dataset.

    Parameters:
        data_dir (str): Location of the dataset.
        name (str): Name of the dataset.
        run_paths (dict): Dictionary containing all the required paths.

    """

    # Get the list of filepaths of the images and label arrays
    (
        train_image_files,
        train_labels,
        val_image_files,
        val_labels,
        test_image_files,
        test_labels,
    ) = get_files(name, data_dir)

    run_paths["path_idrid"] = os.path.join(run_paths["path_idrid"], "1.0.0")
    if not os.path.exists(run_paths["path_idrid"]):
        os.mkdir(run_paths["path_idrid"])

    # Write TFrecords.
    train_shards_length = write_tfrecords(train_image_files, train_labels, run_paths)
    val_shards_length = write_tfrecords(
        val_image_files, val_labels, run_paths, type="val"
    )
    test_shards_length = write_tfrecords(
        test_image_files, test_labels, run_paths, type="test"
    )

    # Create metadata for the TFRecords so that they can be loaded using tfds.
    add_to_tfds(run_paths, train_shards_length, val_shards_length, test_shards_length)
