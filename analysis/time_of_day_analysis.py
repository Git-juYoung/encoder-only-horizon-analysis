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

from config import data_config, model_config
from data import load_and_preprocess
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

    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)

    model.load_state_dict(state)
    model.eval()
    return model


def load_test_with_timestamp(path):
    df = pd.read_csv(path, sep=";", decimal=",", low_memory=False)
    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    power = df.iloc[:, 1:].astype("float32")

    df_2011 = df[df["timestamp"] < "2012-01-01"]
    zero_clients = (df_2011.iloc[:, 1:] == 0).all()
    valid_cols = zero_clients.loc[zero_clients == False].index
    power = power[valid_cols]

    mask = (
        (df["timestamp"] >= data_config["test_start"]) &
        (df["timestamp"] <= data_config["test_end"])
    )

    test_power = power.loc[mask].reset_index(drop=True)
    test_time = df.loc[mask, "timestamp"].reset_index(drop=True)

    return test_power, test_time


def compute_hourly_metrics(model, test_df, test_times, horizon, device):

    input_len = int(data_config["input_length"])
    stride = 1

    values = test_df.values
    hours = test_times.dt.hour.values

    T, num_households = values.shape

    mae_sum = np.zeros(24)
    mae_cnt = np.zeros(24)
    gt_sum = np.zeros(24)
    gt_cnt = np.zeros(24)

    with torch.no_grad():

        for t in range(0, T - input_len - horizon, stride):

            x_np = values[t : t + input_len].T
            y_np = values[t + input_len : t + input_len + horizon]
            target_hours = hours[t + input_len : t + input_len + horizon]

            x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(-1).to(device)
            h_id = torch.arange(num_households, device=device)

            preds = model(x, h_id).cpu().numpy()
            abs_err = np.abs(preds - y_np.T)

            for i in range(horizon):
                h = target_hours[i]
                mae_sum[h] += abs_err[:, i].mean()
                mae_cnt[h] += 1
                gt_sum[h] += y_np[i].mean()
                gt_cnt[h] += 1

    hour_mae = np.divide(mae_sum, mae_cnt, out=np.zeros(24), where=mae_cnt > 0)
    hour_gt = np.divide(gt_sum, gt_cnt, out=np.zeros(24), where=gt_cnt > 0)

    return hour_mae, hour_gt


def main():

    out_dir = get_result_dir()
    device = get_device()

    data_path = os.path.join(BASE_DIR, "LD2011_2014.txt")

    test_df, test_times = load_test_with_timestamp(data_path)

    horizons = [96, 672]

    hour_mae_dict = {}
    hour_gt = None

    for h in horizons:
        print(f"\nRunning horizon {h}...")
        model = load_model(device, h)
        hour_mae, hour_gt_temp = compute_hourly_metrics(
            model, test_df, test_times, h, device
        )
        hour_mae_dict[h] = hour_mae
        hour_gt = hour_gt_temp

    peak_hours = np.argsort(hour_gt)[-4:]
    non_peak_hours = [h for h in range(24) if h not in peak_hours]

    rows = []

    for hour in range(24):
        rows.append({
            "hour": hour,
            "mae_96": float(hour_mae_dict[96][hour]),
            "mae_672": float(hour_mae_dict[672][hour]),
            "is_peak": int(hour in peak_hours)
        })

    peak_mae_96 = np.mean(hour_mae_dict[96][peak_hours])
    non_peak_mae_96 = np.mean(hour_mae_dict[96][non_peak_hours])

    peak_mae_672 = np.mean(hour_mae_dict[672][peak_hours])
    non_peak_mae_672 = np.mean(hour_mae_dict[672][non_peak_hours])

    rows.append({"hour": "peak_hours", "mae_96": str(list(peak_hours)), "mae_672": None, "is_peak": None})
    rows.append({"hour": "peak_mae_96", "mae_96": float(peak_mae_96), "mae_672": None, "is_peak": None})
    rows.append({"hour": "non_peak_mae_96", "mae_96": float(non_peak_mae_96), "mae_672": None, "is_peak": None})
    rows.append({"hour": "diff_96", "mae_96": float(peak_mae_96 - non_peak_mae_96), "mae_672": None, "is_peak": None})
    rows.append({"hour": "peak_mae_672", "mae_96": None, "mae_672": float(peak_mae_672), "is_peak": None})
    rows.append({"hour": "non_peak_mae_672", "mae_96": None, "mae_672": float(non_peak_mae_672), "is_peak": None})
    rows.append({"hour": "diff_672", "mae_96": None, "mae_672": float(peak_mae_672 - non_peak_mae_672), "is_peak": None})

    result_df = pd.DataFrame(rows)

    csv_path = os.path.join(out_dir, "time_of_day_analysis.csv")
    result_df.to_csv(csv_path, index=False)

    plt.figure()
    plt.plot(range(24), hour_mae_dict[96], marker="o", label="H=96")
    plt.plot(range(24), hour_mae_dict[672], marker="o", label="H=672")
    plt.xlabel("Hour")
    plt.ylabel("MAE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "time_of_day_mae_curve.png"))
    plt.close()

    print("\n=== Time-of-Day Analysis ===")
    print(result_df.to_string(index=False))
    print("\nSaved to:", csv_path)
    print("Saved plot to:", os.path.join(out_dir, "time_of_day_mae_curve.png"))


if __name__ == "__main__":
    main()
