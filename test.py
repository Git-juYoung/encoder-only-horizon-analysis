import sys
sys.path.insert(0, "src")

import torch

from seed import set_seed
from config import config, train_config
from data import prepare_train_val_test, build_test_dataloader
from model import build_model
from train_utils import get_device
from evaluate import evaluate_model


def main():

    set_seed()

    device = get_device()

    HORIZON = config["horizon"]
    USE_ID = config["use_id_embedding"]

    _, _, test_dataset = prepare_train_val_test(config)

    test_loader = build_test_dataloader(
        test_dataset=test_dataset,
        batch_size=train_config["batch_size"],
        num_workers=train_config["num_workers"],
        pin_memory=train_config["pin_memory"],
    )

    model = build_model(config).to(device)

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
