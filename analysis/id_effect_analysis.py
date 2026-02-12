import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from config import data_config, model_config, train_config
from data import load_and_preprocess, build_test_dataloader
from dataset import ElectricityDataset
from model import EncoderOnlyTransformer
from train_utils import get_device
from evaluate import compute_metrics


def evaluate(model, loader, device, permute_id=False):
    model.eval()

    preds_list = []
    targets_list = []

    with torch.no_grad():
        for x, y, h_id in loader:
            x = x.to(device)
            y = y.to(device)
            h_id = h_id.to(device)

            if permute_id:
                idx = torch.randperm(h_id.size(0))
                h_id = h_id[idx]

            outputs = model(x, h_id)

            preds_list.append(outputs.cpu())
            targets_list.append(y.cpu())

    preds = torch.cat(preds_list)
    targets = torch.cat(targets_list)

    metrics = compute_metrics(preds, targets)
    return metrics


def main():
    device = get_device()

    result_dir = os.path.join(BASE_DIR, "analysis", "id_effect_results")
    os.makedirs(result_dir, exist_ok=True)

    print("Loading data...")
    data_path = os.path.join(BASE_DIR, "LD2011_2014.txt")
    train, val, test, mean, std = load_and_preprocess(data_path)

    test_dataset = ElectricityDataset(
        test.values,
        data_config["input_length"],
        96,
        data_config["stride"],
    )

    test_loader = build_test_dataloader(
        test_dataset,
        train_config["batch_size"],
        train_config["num_workers"],
        train_config["pin_memory"],
    )

    model_config["use_id_embedding"] = False
    model_config["horizon"] = 96

    noid_model = EncoderOnlyTransformer(model_config).to(device)
    noid_model.load_state_dict(
        torch.load(os.path.join(BASE_DIR, "models", "h96_noid.pt"),
                   map_location=device)
    )

    noid_metrics = evaluate(noid_model, test_loader, device)

    model_config["use_id_embedding"] = True
    model_config["num_households"] = test.shape[1]
    
    id_model = EncoderOnlyTransformer(model_config).to(device)
    id_model.load_state_dict(
        torch.load(os.path.join(BASE_DIR, "models", "h96_id.pt"),
                   map_location=device)
    )

    id_metrics = evaluate(id_model, test_loader, device)

    perm_metrics = evaluate(id_model, test_loader, device, permute_id=True)

    delta_id_vs_noid = id_metrics["mae"] - noid_metrics["mae"]
    delta_perm = perm_metrics["mae"] - id_metrics["mae"]

    result_df = pd.DataFrame([
        {
            "model": "no_id",
            "mse": noid_metrics["mse"],
            "mae": noid_metrics["mae"],
        },
        {
            "model": "id",
            "mse": id_metrics["mse"],
            "mae": id_metrics["mae"],
        },
        {
            "model": "id_permuted",
            "mse": perm_metrics["mse"],
            "mae": perm_metrics["mae"],
        },
    ])

    result_df.to_csv(
        os.path.join(result_dir, "id_effect_summary.csv"),
        index=False
    )

    plt.figure()
    plt.bar(result_df["model"], result_df["mae"])
    plt.ylabel("MAE")
    plt.title("ID Effect Comparison (H=96)")
    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, "id_effect_mae.png"))
    plt.close()

    print("\n=== ID Effect Summary (Horizon=96) ===")
    print(f"No-ID  | MSE: {noid_metrics['mse']:.6f} | MAE: {noid_metrics['mae']:.6f}")
    print(f"ID     | MSE: {id_metrics['mse']:.6f} | MAE: {id_metrics['mae']:.6f}")
    print(f"Perm   | MSE: {perm_metrics['mse']:.6f} | MAE: {perm_metrics['mae']:.6f}")

    print("\nΔ(ID - NoID) MAE:", round(delta_id_vs_noid, 6))
    print("Δ(Perm - ID) MAE:", round(delta_perm, 6))

    print("\nSaved to:", result_dir)


if __name__ == "__main__":
    main()
