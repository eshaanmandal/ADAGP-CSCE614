import torch
from tqdm import tqdm
from validate import validate_baseline

def train_baseline(model, dataloader, val_loader, optimizer, criterion, device, scheduler=None):
    model.train()
    running_loss = 0.0
    for data, labels in tqdm(dataloader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        avg_training_loss = running_loss / len(dataloader)
        val_loss, val_accuracy = validate_baseline(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step(val_loss)
    return avg_training_loss, val_loss, val_accuracy
