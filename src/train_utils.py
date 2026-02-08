import torch
import torch.nn as nn
import torch.optim as optim


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_criterion():
    return nn.MSELoss()


def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def build_optimizer(model, train_config):
    return optim.AdamW(
        model.parameters(),
        lr=train_config["lr"],
        weight_decay=train_config["weight_decay"],
    )


def build_scheduler(optimizer, train_config):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=train_config["factor"],
        patience=train_config["scheduler_patience"],
        verbose=True,
    )
