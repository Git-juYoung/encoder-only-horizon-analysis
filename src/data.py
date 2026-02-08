import pandas as pd
import numpy as np
from config import data_config
from torch.utils.data import DataLoader


def load_and_preprocess(path):

    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        low_memory=False
    )

    df.rename(columns={df.columns[0]: "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    power = df.iloc[:, 1:].astype("float32")

    df_2011 = df[df["timestamp"] < "2012-01-01"]
    zero_clients = (df_2011.iloc[:, 1:] == 0).all()

    valid_columns = zero_clients.loc[zero_clients == False].index
    power = power[valid_columns]

    train_mask = (df["timestamp"] >= data_config["train_start"]) &
                 (df["timestamp"] <= data_config["train_end"])

    val_mask = (df["timestamp"] >= data_config["val_start"]) &
               (df["timestamp"] <= data_config["val_end"])

    test_mask = (df["timestamp"] >= data_config["test_start"]) &
                (df["timestamp"] <= data_config["test_end"])

    train = power[train_mask]
    val = power[val_mask]
    test = power[test_mask]

    mean = train.mean()
    std = train.std()

    train = (train - mean) / std
    val = (val - mean) / std
    test = (test - mean) / std

    return train, val, test, mean, std


def build_train_val_dataloaders(
    train_dataset,
    val_dataset,
    batch_size,
    num_workers,
    pin_memory,
):
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def build_test_dataloader(
    test_dataset,
    batch_size,
    num_workers,
    pin_memory,
):
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return test_loader
