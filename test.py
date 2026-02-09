import sys
sys.path.insert(0, "src")

import torch

from seed import set_seed
from config import data_config, model_config, train_config
from data import prepare_train_val_test, build_test_dataloader
from model import build_model
from train_utils import get_device
from evaluate import evaluate_model


def main():

    set_seed()
    device = get_device()

    HORIZON = model_config["horizon"]
    USE_ID = model_config["use_id_embedding"]

    print(f"\n[Test Setting] Horizon: {HORIZON} | ID Embedding: {USE_ID}")

    _, _, test_dataset = prepare_train_val_test(data_config)
    model_config["num_households"] = test_dataset.data.shape[1]
    
    test_loader = build_test_dataloader(
        test_dataset=test_dataset,
        batch_size=train_config["batch_size"],
        num_workers=train_config["num_workers"],
        pin_memory=train_config["pin_memory"],
    )

    model = build_model(model_config).to(device)

    ckpt_path = train_config["save_path"]

    model.load_state_dict(
        torch.load(ckpt_path, map_location=device)
    )

    model.eval()

    results = evaluate_model(
        model=model,
        loader=test_loader,
        device=device,
    )

    print("\n===== Test Results =====")
    print(f"MAE  : {results['mae']:.6f}")
    print(f"RMSE : {results['rmse']:.6f}")


if __name__ == "__main__":
    main()
