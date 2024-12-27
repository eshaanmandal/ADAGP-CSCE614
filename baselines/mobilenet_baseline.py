import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"Setting seed of {seed}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for pre-trained models
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Dataset to be used: CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Dataloader for loading dataset into the network
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

# Use MobileNet instead of VGG
mobilenet = models.mobilenet_v2(pretrained=False)  # Use pretrained weights
mobilenet.classifier[1] = nn.Linear(mobilenet.last_channel, 10)  # Update for 10 classes (CIFAR-10)
mobilenet = mobilenet.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(mobilenet.parameters(), lr=0.001, momentum=0.9)  # Using SGD with momentum
scheduler_model = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

def evaluate_model_loss(model, dataloader, criterion, device="cuda"):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def train_model(
        model,
        trainloader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=10,
        starting_epoch=0,
        device="cuda",
        seed=42):

    for epoch in range(starting_epoch, starting_epoch + num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        t1 = time.time()
        for inputs, labels in tqdm(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        t2 = time.time()

        per_epoch_time = t2 - t1
        print(f"Took {per_epoch_time} seconds/epoch")

        avg_train_loss = running_loss / len(trainloader)

        scheduler.step(avg_train_loss)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}')

    return model

# Train the model
start_time = time.time()

train_model(
    mobilenet,
    trainloader,
    criterion,
    optimizer,
    scheduler_model,
    num_epochs=2,
    starting_epoch=0,
    device=device
)

end_time = time.time()
print(f"It took around {end_time - start_time} seconds for training the model.")

# Evaluate the model on the test set after training
test_loss, test_accuracy = evaluate_model_loss(mobilenet, testloader, criterion, device)
print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
