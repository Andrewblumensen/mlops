import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from model import MyAwesomeModel, MyCNNModel

# Load the trained model
#model = MyAwesomeModel()
#model.load_state_dict(torch.load('trained_FCNN_model.pth'))
model = MyCNNModel()
model.load_state_dict(torch.load('trained_CNN_model.pth'))
model.eval()

# Load test data
test_images_file_path = '/Users/andrewblumensen/Documents/mlops/Day1/data/test_images.pt'
test_images_state_dict = torch.load(test_images_file_path)

test_target_file_path = '/Users/andrewblumensen/Documents/mlops/Day1/data/test_target.pt'
test_target_state_dict = torch.load(test_target_file_path)

test_images_tensor = test_images_state_dict
test_target_tensor = test_target_state_dict

# Perform inference on test data
with torch.no_grad():
    outputs = model(test_images_tensor)

# Get the predicted labels
_, predicted_labels = torch.max(outputs, 1)

# Print the predicted labels
print("Predicted Labels:", predicted_labels.numpy())

# Calculate accuracy
correct_predictions = (predicted_labels == test_target_tensor).sum().item()
total_samples = test_target_tensor.size(0)
accuracy = correct_predictions / total_samples

print(f"Accuracy: {accuracy * 100:.2f}%")

# Plot the five first images along with predictions
fig, axs = plt.subplots(1, 5, figsize=(15, 3))

for i in range(5):
    image = test_images_tensor[i].view(28, 28).numpy()
    label = predicted_labels[i].item()

    axs[i].imshow(image, cmap='gray')
    axs[i].set_title(f'Prediction: {label}')
    axs[i].axis('off')

plt.show()
