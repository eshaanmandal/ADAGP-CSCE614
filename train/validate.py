import torch

def validate_baseline(model, dataloader, criterion, device):
    model.eval()
    validation_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, labels in dataloader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            # Example for classification accuracy
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    accuracy = correct / len(dataloader.dataset)
    return validation_loss / len(dataloader), accuracy
