import torch
from data import get_dataloader
from models import get_model
from train import train_baseline, validate_baseline
from utils import Config, save_checkpoint, load_checkpoint
import random
import numpy as np
# from utils.config import Config  # Import a configuration file if you have one
def set_seed(seed):
    """
    Fixes the random seed for reproducibility.
    
    Args:
        seed (int): The seed value to set for random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures deterministic behavior


def main():
    # Load configuration settings (if using a Config class or dictionary for hyperparameters)
    config = Config()  # Adjust as needed
    set_seed(config.seed)
    # Set device
    device = config.device
    print(f"Using {device}")

    # Initialize the model
    model = get_model(model_name=config.model_name, pretrained=config.pretrained).to(device)

    # Initialize optimizer and loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=config.mode, factor=config.factor, patience=config.patience, verbose=config.verbose)
    s_epoch=0
    if config.use_checkpoint:
        s_epoch, _ = load_checkpoint(model, optimizer,file_path=f'./checkpoints/resnet50_epoch_29.pth', scheduler=scheduler)
    # Load data
    train_loader, val_loader, test_loader = get_dataloader(batch_size=config.batch_size)

    # Run training and validation loops
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+s_epoch+1}/{config.num_epochs}")

        # Training
        train_loss, val_loss, val_accuracy = train_baseline(model, train_loader, val_loader, optimizer, criterion, device)

        # # Validation
        # val_loss, val_accuracy = validate_baseline(model, val_loader, criterion, device)

        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy*100}%")

        # Optional: Save checkpoint
        save_checkpoint(model, optimizer, epoch, val_loss, file_path=f'./checkpoints/{config.model_name}_epoch_{epoch+s_epoch+1}.pth', scheduler=scheduler)

    print("checking the accuracy on test set")
    test_loss, test_accuracy = validate_baseline(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy*100}%")

if __name__ == "__main__":
    main()
