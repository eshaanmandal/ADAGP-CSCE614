import torch
from data import get_dataloader
from models import get_model
from train import train_baseline, validate_baseline
from utils import Config, save_checkpoint, load_checkpoint
# from utils.config import Config  # Import a configuration file if you have one


def main():
    # Load configuration settings (if using a Config class or dictionary for hyperparameters)
    config = Config()  # Adjust as needed

    # Set device
    device = config.device

    # Initialize the model
    model = get_model(model_name=config.model_name, pretrained=config.pretrained).to(device)

    # Initialize optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=config.mode, factor=config.factor, patience=config.patience, verbose=config.verbose)
    if config.use_checkpoint:
        _, _ = load_checkpoint(model, optimizer,file_path=f'./checkpoints/{config.model_name}_epoch_{epoch}.pth', scheduler=scheduler)
    # Load data
    train_loader, val_loader = get_dataloader(batch_size=config.batch_size)

    # Run training and validation loops
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+1}/{config.num_epochs}")

        # Training
        train_loss = train_baseline(model, train_loader, optimizer, criterion, device)

        # Validation
        val_loss, val_accuracy = validate_baseline(model, val_loader, criterion, device)

        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        # Optional: Save checkpoint
        save_checkpoint(model, optimizer, epoch, val_loss, file_path=f'./checkpoints/{config.model_name}_epoch_{epoch}.pth', scheduler=scheduler)

if __name__ == "__main__":
    main()
