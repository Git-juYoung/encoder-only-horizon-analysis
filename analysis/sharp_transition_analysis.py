import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import copy
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from config import data_config, model_config, train_config
from data import load_and_preprocess, build_test_dataloader
from dataset import ElectricityDataset
from model import EncoderOnlyTransformer
from train_utils import get_device


def get_result_dir():
    out_dir = os.path.join(BASE_DIR, "analysis", "structural_results")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def load_model(device, horizon):
    cfg = copy.deepcopy(model_config)
    cfg["use_id_embedding"] = False
    cfg["horizon"] = horizon
    model = EncoderOnlyTransformer(cfg).to(device)
    ckpt = os.path.join(BASE_DIR, "models", f"h{horizon}_noid.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


def build_loader(test_df, horizon):
    dataset = ElectricityDataset(
        test_df.values,
        data_config["input_length"],
        horizon,
        horizon,
    )
    loader = build_test_dataloader(
        dataset,
        train_config["batch_size"],
        train_config["num_workers"],
        train_config["pin_memory"],
    )
    return loader


def main():
    out_dir = get_result_dir()
    device = get_device()
    horizons = [96, 192, 672]

    data_path = os.path.join(BASE_DIR, "LD2011_2014.txt")
    train_df, val_df, test_df, mean, std = load_and_preprocess(data_path)

    results = []

    for h in horizons:
        loader = build_loader(test_df, h)
        model = load_model(device, h)

        sharp_errors = []
        normal_errors = []

        model.eval()
        with torch.no_grad():
            for x, y, h_id in loader:
                x = x.to(device)
                y = y.to(device)

                preds = model(x, h_id)
                abs_error = torch.abs(preds - y)

                delta_y = torch.abs(y[:, 1:] - y[:, :-1])
                threshold = torch.quantile(delta_y.flatten(), 0.8)

                sharp_mask = delta_y >= threshold

                sharp_mask = torch.cat(
                    [sharp_mask, torch.zeros_like(sharp_mask[:, :1])],
                    dim=1
                )

                sharp_errors.append(abs_error[sharp_mask].cpu())
                normal_errors.append(abs_error[~sharp_mask].cpu())

        sharp_errors = torch.cat(sharp_errors).numpy()
        normal_errors = torch.cat(normal_errors).numpy()

        mae_sharp = float(np.mean(sharp_errors))
        mae_normal = float(np.mean(normal_errors))
        penalty = mae_sharp - mae_normal

        results.append({
            "horizon": h,
            "mae_sharp": mae_sharp,
            "mae_normal": mae_normal,
            "sharp_penalty": penalty
        })

    result_df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "sharp_transition_summary.csv")
    result_df.to_csv(csv_path, index=False)

    plt.figure()
    plt.plot(result_df["horizon"], result_df["mae_sharp"], marker="o", label="sharp")
    plt.plot(result_df["horizon"], result_df["mae_normal"], marker="o", label="normal")
    plt.xlabel("horizon")
    plt.ylabel("MAE")
    plt.title("Sharp vs Normal MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sharp_transition_curve.png"))
    plt.close()

    print("\n=== Sharp Transition Analysis Summary ===")
    print(result_df.to_string(index=False))
    print("\nSaved to:", csv_path)


if __name__ == "__main__":
    main()
