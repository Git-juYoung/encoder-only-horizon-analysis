import os
import sys
import torch
import numpy as np

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
    households_list = []

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
            households_list.append(h_id.cpu())

    preds = torch.cat(preds_list)
    targets = torch.cat(targets_list)
    households = torch.cat(households_list)

    metrics = compute_metrics(preds, targets)

    return metrics, preds, targets, households


def household_error(preds, targets, households):
    sample_mae = torch.abs(preds - targets).mean(dim=1)

    result = {}
    for h in torch.unique(households):
        mask = households == h
        result[int(h.item())] = sample_mae[mask].mean().item()

    return result


def compute_gap(log_path):
    train_losses = []
    val_losses = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            if "train_loss:" in line and "val_loss:" in line:

                train_part = line.split("train_loss:")[1]
                train_value = train_part.split("|")[0].strip()
                train_losses.append(float(train_value))

                val_part = line.split("val_loss:")[1]
                val_value = val_part.split("|")[0].strip()
                val_losses.append(float(val_value))

    n = min(len(train_losses), len(val_losses))
    gaps = [val_losses[i] - train_losses[i] for i in range(n)]

    return np.mean(gaps)


def main():
    device = get_device()

    print("Loading data...")
    data_path = os.path.join(BASE_DIR, "LD2011_2014.txt")
    train, val, test, mean, std = load_and_preprocess(data_path)

    test_dataset = ElectricityDataset(
        test.values,
        data_config["input_length"],
        data_config["horizon"],
        data_config["stride"],
    )

    test_loader = build_test_dataloader(
        test_dataset,
        train_config["batch_size"],
        train_config["num_workers"],
        train_config["pin_memory"],
    )

    print("Loading model...")
    model_path = os.path.join(BASE_DIR, train_config["save_path"])
    model = EncoderOnlyTransformer(model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print("\n=== Normal Evaluation ===")
    normal_metrics, preds, targets, households = evaluate(
        model, test_loader, device
    )
    print(normal_metrics)

    print("\n=== Permutation Evaluation ===")
    perm_metrics, _, _, _ = evaluate(
        model, test_loader, device, permute_id=True
    )
    print(perm_metrics)

    print("\nPermutation Impact (MSE diff):",
          perm_metrics["mse"] - normal_metrics["mse"])

    print("\n=== Household MAE (first 10) ===")
    h_errors = household_error(preds, targets, households)
    for i, (h, v) in enumerate(h_errors.items()):
        print(f"Household {h}: {v:.4f}")
        if i == 9:
            break

    print("\n=== Train-Val Gap ===")
    log_path = os.path.join(BASE_DIR, "log", "h96_id.txt")
    gap = compute_gap(log_path)
    print("Average gap:", gap)


if __name__ == "__main__":
    main()
