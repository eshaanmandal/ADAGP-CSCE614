import torch
from tqdm import tqdm

def train_baseline(model, dataloader, optimizer, criterion, device, scheduler=None):
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
        if scheduler is not None:
            scheduler.step(avg_training_loss)
    return avg_training_loss
