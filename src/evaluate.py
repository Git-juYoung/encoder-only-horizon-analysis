import torch


def compute_metrics(preds, targets):

    mse = torch.mean((preds - targets) ** 2)
    mae = torch.mean(torch.abs(preds - targets))
    rmse = torch.sqrt(mse)

    return {
        "mse": mse.item(),
        "mae": mae.item(),
        "rmse": rmse.item(),
    }
