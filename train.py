import sys
sys.path.insert(0, "src")

import os
import torch
import wandb

from seed import set_seed
from config import data_config, model_config, train_config
from data import load_and_preprocess, build_train_val_dataloaders
from dataset import ElectricityDataset
from model import EncoderOnlyTransformer
from train_utils import get_device, build_criterion, build_optimizer, build_scheduler
from engine import train_one_epoch, validate_one_epoch
from early_stopping import EarlyStopping


def main():
    set_seed()
    device = get_device()

    train_df, val_df, _, _, _ = load_and_preprocess("LD2011_2014.txt")

    train_data = train_df.values
    val_data = val_df.values

    train_dataset = ElectricityDataset(
        train_data,
        data_config["input_length"],
        data_config["horizon"],
        data_config["stride"],
    )

    val_dataset = ElectricityDataset(
        val_data,
        data_config["input_length"],
        data_config["horizon"],
        data_config["stride"],
    )

    train_loader, val_loader = build_train_val_dataloaders(
        train_dataset,
        val_dataset,
        train_config["batch_size"],
        train_config["num_workers"],
        train_config["pin_memory"],
    )

    model_config["num_households"] = train_data.shape[1]

    model = EncoderOnlyTransformer(model_config).to(device)

    criterion = build_criterion()
    optimizer = build_optimizer(model, train_config)
    scheduler = build_scheduler(optimizer, train_config)

    os.makedirs(os.path.dirname(train_config["save_path"]), exist_ok=True)

    early_stopper = EarlyStopping(
        patience=train_config["early_patience"],
        save_path=train_config["save_path"],
    )

    wandb.init(
        project="electricity_transformer",
        name=f"h{model_config['horizon']}_{'id' if model_config['use_id_embedding'] else 'noid'}",
        config={**data_config, **model_config, **train_config},
    )

    for epoch in range(1, train_config["epochs"] + 1):

        train_loss, train_time = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, train_config["epochs"]
        )

        val_loss, val_time = validate_one_epoch(
            model, val_loader, criterion, device, epoch, train_config["epochs"]
        )

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch}/{train_config['epochs']} | "
            f"train_loss: {train_loss:.6f} | "
            f"val_loss: {val_loss:.6f} | "
            f"train_time: {train_time:.1f}s | "
            f"val_time: {val_time:.1f}s"
        )

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        })

        if early_stopper.step(val_loss, model):
            break

    wandb.finish()


if __name__ == "__main__":
    main()