import torch
class Config:
    # General settings
    seed = 42  # Random seed for reproducibility
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cuda"

    # Data parameters
    data_dir = "./data"  # Directory for storing data
    batch_size = 64  # Batch size for training and validation

    # Model parameters
    model_name = 'resnet50'  # Model architecture, e.g., 'resnet50', 'vgg16'
    pretrained = False  # Whether to use a pretrained model

    # Training parameters
    num_epochs = 100  # Number of training epochs
    lr = 0.001  # Learning rate for optimizer
    weight_decay = 1e-4  # Weight decay (L2 regularization)
    mode='min'
    factor=0.1
    patience=5
    verbose=True

    # Checkpoint parameters
    checkpoint_dir = "./checkpoints"  # Directory to save model checkpoints
    save_freq = 1  # Frequency (in epochs) to save checkpoints
    use_checkpoint=True
    
    # Logging parameters
    log_freq = 10  # Frequency of logging during training

# Example usage
# config = Config()
# print(config.batch_size)  # Access any attribute like this
