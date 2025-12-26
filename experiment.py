import json
import glob
import torch

from src.validation import *
from src.data.train_test_split import get_divided_train_test

if __name__ == "__main__":

    results_dir = "results"
    exp_num = len(glob.glob(results_dir+"/reconstruction_errs_*.json"))+1
    save_to = f"{results_dir}/reconstruction_errs_unhealthy.json"
    print(f"will save to {save_to}")

    _, unhealthy_files = get_divided_train_test(1, 'unhealthy')
    # with open("trained_models/healthy_test_cells.json") as file:
    #     healthy_files = json.load(file)

    device = "mps" if torch.mps.is_available() else "cpu"
    print("Using device:", device)

    store_reconstruction_errors(
        model_filename="trained_models/3epoch_0.8healthy_500window_300step.pkl",
        filenames=unhealthy_files,
        device=device,
        save_to=save_to
    )