import sys
sys.path.insert(0, "src")

import torch

from seed import set_seed
from config import data_config, model_config, train_config
from data import load_and_preprocess, build_test_dataloader
from dataset import ElectricityDataset
from model import EncoderOnlyTransformer
from train_utils import get_device
from evaluate import compute_metrics


def main():
    set_seed()
    device = get_device()

    _, _, test_df, _, _ = load_and_preprocess("LD2011_2014.txt")
    test_data = test_df.values

    test_dataset = ElectricityDataset(
        test_data,
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

    model_config["num_households"] = test_data.shape[1]

    model = EncoderOnlyTransformer(model_config).to(device)

    model.load_state_dict(
        torch.load(train_config["save_path"], map_location=device)
    )

    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y, h_id in test_loader:
            x = x.to(device)
            y = y.to(device)
            h_id = h_id.to(device)

            preds = model(x, h_id)

            all_preds.append(preds)
            all_targets.append(y)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(all_preds, all_targets)

    print("\n===== Test Results =====")
    print(f"MSE  : {metrics['mse']:.6f}")
    print(f"MAE  : {metrics['mae']:.6f}")
    print(f"RMSE : {metrics['rmse']:.6f}")


if __name__ == "__main__":
    main()
