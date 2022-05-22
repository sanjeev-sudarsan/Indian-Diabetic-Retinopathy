"""
Keep track of utils
"""
import os
import datetime


def gen_run_folder(path_model_id=datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")):
    """Create directory for the run, log files and checkpoints.

    Returns:
        (dict): Dictionary containing names of directories and their paths, paths are pathlib.Path.
    """

    run_paths = {}
    path_model_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "experiments")
    )
    run_paths["path_model_id"] = os.path.join(path_model_root, path_model_id)
    run_paths["path_datasets"] = os.path.abspath(
        os.path.join(os.path.dirname(__file__), os.pardir, "datasets")
    )
    run_paths["path_idrid"] = os.path.join(run_paths["path_datasets"], "idrid_dataset")

    # directories
    run_paths["path_summaries"] = os.path.join(run_paths["path_model_id"], "summaries")
    run_paths["path_weights"] = os.path.join(run_paths["path_model_id"], "weights")
    run_paths["path_logs"] = os.path.join(run_paths["path_model_id"], "logs")
    run_paths["path_ckpts"] = os.path.join(run_paths["path_model_id"], "checkpoints")
    run_paths["path_gin"] = os.path.join(run_paths["path_model_id"], "gins")
    run_paths["path_train_summary"] = os.path.join(run_paths["path_summaries"], "train")
    run_paths["path_val_summary"] = os.path.join(run_paths["path_summaries"], "val")
    run_paths["path_test_summary"] = os.path.join(run_paths["path_summaries"], "test")
    run_paths["path_vis_summary"] = os.path.join(run_paths["path_summaries"], "vis")

    # files
    run_paths["file_run_log"] = os.path.join(run_paths["path_logs"], "run.log")
    run_paths["file_training_log"] = os.path.join(
        run_paths["path_logs"], "training.log"
    )
    run_paths["file_evaluation_log"] = os.path.join(
        run_paths["path_logs"], "evaluation.log"
    )
    run_paths["file_gin"] = os.path.join(run_paths["path_gin"], "config_operative.gin")

    # Create folders
    for name, path in run_paths.items():
        if ("path_" in name) and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

    # Create files
    for name, path in run_paths.items():
        if ("file_" in name) and not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8"):
                pass  # atm file creation is sufficient

    return run_paths


def save_config(path_gin, config):
    """
    function to write and save config file
    """
    with open(path_gin, "w", encoding="utf-8") as f_config:
        f_config.write(config)
