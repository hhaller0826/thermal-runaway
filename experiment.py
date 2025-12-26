import json
import glob
import torch

from src.validation import *
from src.data.train_test_split import get_divided_train_test

if __name__ == "__main__":

    results_dir = "results"
    exp_num = len(glob.glob(results_dir+"/reconstruction_errs_*.json"))

    _, unhealthy_files = get_divided_train_test(1, 'unhealthy')

    device = "mps" if torch.mps.is_available() else "cpu"
    print("Using device:", device)

    store_reconstruction_errors(
        model_filename="trained_models/3epoch_0.8healthy_500window_300step.pkl",
        filenames=unhealthy_files,
        device=device,
        save_to=f"{results_dir}/reconstruction_errs_{exp_num}.json"
    )