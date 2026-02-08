from tqdm import tqdm
import time
import torch


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    epoch,
    num_epochs,
):
    model.train()

    start_time = time.time()

    total_loss = 0.0
    total_batches = 0

    train_bar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{num_epochs} [Train]",
        leave=False
    )

    for x, y, h_id in train_bar:

        x = x.to(device)
        y = y.to(device)
        h_id = h_id.to(device)

        optimizer.zero_grad()

        outputs = model(x, h_id)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

    epoch_time = time.time() - start_time
    avg_loss = total_loss / total_batches

    return avg_loss, epoch_time


def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    epoch,
    num_epochs,
):
    model.eval()
    start_time = time.time()
    
    total_loss = 0.0
    total_batches = 0

    val_bar = tqdm(
        loader,
        desc=f"Epoch {epoch}/{num_epochs} [Val]",
        leave=False
    )

    with torch.no_grad():
        for x, y, h_id in val_bar:

            x = x.to(device)
            y = y.to(device)
            h_id = h_id.to(device)

            outputs = model(x, h_id)

            loss = criterion(outputs, y)

            total_loss += loss.item()
            total_batches += 1

    epoch_time = time.time() - start_time
    avg_loss = total_loss / total_batches
    return avg_loss, epoch_time
