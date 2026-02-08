import os
import sys
sys.path.insert(0, "src")

import torch
import wandb

from seed import set_seed
from config import config
from data import prepare_train_val_test, build_train_val_dataloaders
from model import build_model
from engine import train_one_epoch, validate_one_epoch
from evaluate import evaluate_model
from train_utils import get_device, build_optimizer, build_scheduler
from early_stopping import EarlyStopping


def main():

    set_seed()

    HORIZON = config["horizon"]
    USE_ID = config["use_id_embedding"]

    WANDB_PROJECT = "electricity_transformer"
    WANDB_RUN_NAME = f"h{HORIZON}_{'id' if USE_ID else 'noid'}"

    run = wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config=config,
    )

    device = get_device()

    save_dir = os.path.join("models", f"h{HORIZON}")
    os.makedirs(save_dir, exist_ok=True)

    train_dataset, val_dataset = prepare_train_val_test(config)

    train_loader, val_loader = build_train_val_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
    )

    model = build_model(config).to(device)

    optimizer = build_optimizer(
        model,
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    scheduler = build_scheduler(
        optimizer,
        mode="min",
        factor=config["scheduler"]["factor"],
        patience=config["scheduler"]["patience"],
    )

    criterion = torch.nn.MSELoss()

    early_stopper = EarlyStopping(
        patience=config["early_stopping"]["patience"],
        save_path=save_path,
    )

    num_epochs = config["num_epochs"]

    for epoch in range(1, num_epochs + 1):

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
        )

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
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
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
        f"MAE: {final_metrics['mae']:.4f} | "
        f"RMSE: {final_metrics['rmse']:.4f}"
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
