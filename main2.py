import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from model import MyAwesomeModel
from model import MyCNNModel  # Assuming you save the CNN model in cnn_model.py
from data import mnist
import matplotlib.pyplot as plt
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Train FCNN or CNN on MNIST data')
parser.add_argument('--model', choices=['FCNN', 'CNN'], required=True, help='Choose between FCNN or CNN')
args = parser.parse_args()

# Load your data using the mnist function from data.py
train_images_tensor, train_target_tensor = mnist()

# Split the data into training and validation sets
split_index = 5000
train_images_tensor, val_images_tensor = (
    train_images_tensor[split_index:],
    train_images_tensor[:split_index],
)
train_target_tensor, val_target_tensor = (
    train_target_tensor[split_index:],
    train_target_tensor[:split_index],
)

# Choose the model based on the command-line argument
if args.model == 'FCNN':
    model = MyAwesomeModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
elif args.model == 'CNN':
    model = MyCNNModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
else:
    raise ValueError('Invalid model choice. Use --model FCNN or --model CNN')

# Combine data into DataLoaders for training and validation
train_dataset = TensorDataset(train_images_tensor, train_target_tensor)
val_dataset = TensorDataset(val_images_tensor, val_target_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Training loop
epochs = 20

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(epochs):
    # Training
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = correct_train / total_train
    train_losses.append(running_loss / len(train_dataloader))
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    running_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_dataloader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = correct_val / total_val
    val_losses.append(running_loss / len(val_dataloader))
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Model: {args.model}, "
        f"Train Loss: {train_losses[-1]:.4f}, "
        f"Train Accuracy: {train_accuracy:.4f}, "
        f"Val Loss: {val_losses[-1]:.4f}, "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

print("Training finished!")

# Plot training and validation accuracy curves
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, epochs + 1), val_accuracies, label='Validation Accuracy')
plt.title(f'Training and Validation Accuracy Curves for {args.model}')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the trained model if needed
torch.save(model.state_dict(), f'trained_{args.model}_model.pth')

