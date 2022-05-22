import logging
import tensorflow as tf


def set_loggers(paths=None, logging_level=0, b_stream=False, b_debug=False):

    # std. logger
    logger = logging.getLogger()
    logger.setLevel(logging_level)

    # tf logger
    logger_tf = tf.get_logger()
    logger_tf.setLevel(logging_level)

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler = logging.FileHandler(paths["file_run_log"])
    file_handler.setFormatter(formatter)
    logger.addHandler(hdlr=file_handler)
    logger_tf.addHandler(file_handler)

    # Training logger
    training_logger = logging.getLogger("training")
    training_logger.setLevel(logging_level)
    train_handler = logging.FileHandler(paths["file_training_log"])
    train_handler.setFormatter(formatter)
    training_logger.addHandler(hdlr=train_handler)

    evaluation_logger = logging.getLogger("evaluation")
    evaluation_logger.setLevel(logging_level)
    evaluation_handler = logging.FileHandler(paths["file_evaluation_log"])
    evaluation_handler.setFormatter(formatter)
    evaluation_logger.addHandler(hdlr=evaluation_handler)

    # plot to console
    if b_stream:
        stream_handler = logging.StreamHandler()
        logger.addHandler(stream_handler)

    if b_debug:
        tf.debugging.set_log_device_placement(False)
