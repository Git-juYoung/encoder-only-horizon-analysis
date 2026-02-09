import os
import sys
sys.path.insert(0, "src")

import torch
import wandb

from seed import set_seed
from config import data_config, model_config, train_config
from data import prepare_train_val_test, build_train_val_dataloaders
from model import build_model
from engine import train_one_epoch, validate_one_epoch
from evaluate import evaluate_model
from train_utils import get_device, build_optimizer, build_scheduler
from early_stopping import EarlyStopping


def main():

    set_seed()
    device = get_device()

    HORIZON = model_config["horizon"]
    USE_ID = model_config["use_id_embedding"]

    WANDB_PROJECT = "electricity_transformer"
    WANDB_RUN_NAME = f"h{HORIZON}_{'id' if USE_ID else 'noid'}"

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            **data_config,
            **model_config,
            **train_config,
        },
    )

    train_dataset, val_dataset = prepare_train_val_test(data_config)
    model_config["num_households"] = train_dataset.data.shape[1]
    
    train_loader, val_loader = build_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=train_config["batch_size"],
        num_workers=train_config["num_workers"],
        pin_memory=train_config["pin_memory"],
    )

    model = build_model(model_config).to(device)

    optimizer = build_optimizer(model, train_config)

    scheduler = build_scheduler(optimizer, train_config)

    criterion = torch.nn.MSELoss()

    save_path = train_config["save_path"]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    early_stopper = EarlyStopping(
        patience=train_config["early_patience"],
        save_path=save_path,
    )

    num_epochs = train_config["epochs"]

    for epoch in range(1, num_epochs + 1):

        train_loss, _ = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
        )

        val_loss, _ = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            num_epochs=num_epochs,
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_loss,
                "lr": current_lr,
            },
            step=epoch,
        )

        print(
            f"[Epoch {epoch}/{num_epochs}] "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if early_stopper.step(val_loss, model):
            print("Early stopping triggered.")
            break

    print("\nLoading best model...")
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    model.eval()

    final_metrics = evaluate_model(
        model=model,
        loader=val_loader,
        device=device,
    )

    print("\n[Best Model Performance]")
    print(
        f"MAE: {final_metrics['mae']:.6f} | "
        f"RMSE: {final_metrics['rmse']:.6f}"
    )

    wandb.log(
        {
            "best/val_mae": final_metrics["mae"],
            "best/val_rmse": final_metrics["rmse"],
        }
    )

    wandb.finish()

if __name__ == "__main__":
    main()
