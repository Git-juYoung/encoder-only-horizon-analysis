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
from evaluate import compute_metrics


def get_result_dir():
    out_dir = os.path.join(BASE_DIR, "analysis", "structural_results")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


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


def load_model(device, horizon):
    cfg = copy.deepcopy(model_config)
    cfg["use_id_embedding"] = False
    cfg["horizon"] = horizon
    model = EncoderOnlyTransformer(cfg).to(device)
    ckpt = os.path.join(BASE_DIR, "models", f"h{horizon}_noid.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model


def evaluate_full(model, loader, device):
    model.eval()
    preds_list = []
    targets_list = []
    households_list = []

    with torch.no_grad():
        for x, y, h_id in loader:
            x = x.to(device)
            y = y.to(device)
            h_id = h_id.to(device)
            out = model(x, h_id)
            preds_list.append(out.cpu())
            targets_list.append(y.cpu())
            households_list.append(h_id.cpu())

    preds = torch.cat(preds_list)
    targets = torch.cat(targets_list)
    households = torch.cat(households_list)

    metrics = compute_metrics(preds, targets)
    return metrics, preds, targets, households


def household_mae(preds, targets, households):
    sample_mae = torch.abs(preds - targets).mean(dim=1)
    result = {}
    for h in torch.unique(households):
        mask = households == h
        result[int(h.item())] = sample_mae[mask].mean().item()
    return result


def compute_features(test_df, lag=96):
    x = test_df.values
    T, N = x.shape
    rows = []
    for j in range(N):
        s = x[:, j].astype(np.float64)
        var = float(np.var(s))
        sharp = float(np.mean(np.abs(np.diff(s))))
        if T > lag + 1 and np.std(s[:-lag]) > 1e-12 and np.std(s[lag:]) > 1e-12:
            ac = float(np.corrcoef(s[:-lag], s[lag:])[0, 1])
        else:
            ac = np.nan
        rows.append([j, var, sharp, ac])
    return pd.DataFrame(rows, columns=["household_id", "variance", "sharpness", "autocorr"])


def main():
    out_dir = get_result_dir()
    device = get_device()
    horizons = [96, 192, 672]

    data_path = os.path.join(BASE_DIR, "LD2011_2014.txt")
    train_df, val_df, test_df, mean, std = load_and_preprocess(data_path)

    overall_metrics = []
    hh_mae_map = {}

    for h in horizons:
        loader = build_loader(test_df, h)
        model = load_model(device, h)
        metrics, preds, targets, households = evaluate_full(model, loader, device)

        overall_metrics.append((h, metrics["mse"], metrics["mae"], metrics["rmse"]))
        hh_mae_map[h] = household_mae(preds, targets, households)

    overall_df = pd.DataFrame(overall_metrics, columns=["horizon", "mse", "mae", "rmse"])

    plt.figure()
    plt.plot(overall_df["horizon"], overall_df["mae"], marker="o")
    plt.xlabel("horizon")
    plt.ylabel("MAE")
    plt.title("Overall MAE vs Horizon")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "horizon_overall_curve.png"))
    plt.close()

    num_households = test_df.shape[1]
    hh_ids = np.arange(num_households)
    slope_list = []

    for hid in hh_ids:
        e96 = hh_mae_map[96].get(hid, np.nan)
        e672 = hh_mae_map[672].get(hid, np.nan)
        if not np.isnan(e96) and not np.isnan(e672):
            slope = (e672 - e96) / (672 - 96)
            slope_list.append((hid, slope))

    slope_df = pd.DataFrame(slope_list, columns=["household_id", "slope"])
    slopes = slope_df["slope"].dropna()

    slope_mean = slopes.mean()
    slope_std = slopes.std()
    slope_min = slopes.min()
    slope_max = slopes.max()

    plt.figure()
    plt.hist(slopes, bins=50)
    plt.xlabel("slope")
    plt.ylabel("count")
    plt.title("Slope Distribution (96→672)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "horizon_slope_distribution.png"))
    plt.close()

    k = max(1, int(len(slopes) * 0.2))
    top_ids = slope_df.sort_values("slope", ascending=False).head(k)["household_id"]
    bottom_ids = slope_df.sort_values("slope", ascending=True).head(k)["household_id"]

    features = compute_features(test_df).set_index("household_id")

    top_mean = features.loc[top_ids].mean()
    bottom_mean = features.loc[bottom_ids].mean()

    var_top = features["variance"].idxmax()
    var_bottom = features["variance"].idxmin()
    slope_top = slope_df.sort_values("slope", ascending=False).iloc[0]["household_id"]
    slope_bottom = slope_df.sort_values("slope", ascending=True).iloc[0]["household_id"]

    selected_ids = [int(var_top), int(var_bottom), int(slope_top), int(slope_bottom)]

    plt.figure()
    for hid in selected_ids:
        ys = [
            hh_mae_map[96].get(hid, np.nan),
            hh_mae_map[192].get(hid, np.nan),
            hh_mae_map[672].get(hid, np.nan),
        ]
        plt.plot(horizons, ys, marker="o", label=f"hh_{hid}")
    plt.xlabel("horizon")
    plt.ylabel("MAE")
    plt.title("Representative Error Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "horizon_representative_curves.png"))
    plt.close()

    summary_rows = [
        {
            "overall_mae_96": overall_df.loc[overall_df["horizon"] == 96, "mae"].values[0],
            "overall_mae_192": overall_df.loc[overall_df["horizon"] == 192, "mae"].values[0],
            "overall_mae_672": overall_df.loc[overall_df["horizon"] == 672, "mae"].values[0],
            "slope_mean": slope_mean,
            "slope_std": slope_std,
            "slope_min": slope_min,
            "slope_max": slope_max,
            "variance_top20": top_mean["variance"],
            "variance_bottom20": bottom_mean["variance"],
            "sharpness_top20": top_mean["sharpness"],
            "sharpness_bottom20": bottom_mean["sharpness"],
            "autocorr_top20": top_mean["autocorr"],
            "autocorr_bottom20": bottom_mean["autocorr"],
        }
    ]

    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(out_dir, "horizon_analysis_summary.csv")
    summary_df.to_csv(csv_path, index=False)

    print("\n=== Horizon Analysis Summary ===")
    print(summary_df.to_string(index=False))
    print("\nSaved to:", csv_path)


if __name__ == "__main__":
    main()
