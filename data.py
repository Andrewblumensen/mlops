import torch
import matplotlib.pyplot as plt

def mnist():
    # Initialize empty lists to store train and test tensors
    train_images_tensors = []
    train_target_tensors = []

    # Loop through train files
    for i in range(6):
        train_images_file_path = f'/Users/andrewblumensen/Documents/mlops/Day1/data/train_images_{i}.pt'
        train_images_state_dict = torch.load(train_images_file_path)
        train_images_tensors.append(train_images_state_dict)

        train_target_file_path = f'/Users/andrewblumensen/Documents/mlops/Day1/data/train_target_{i}.pt'
        train_target_state_dict = torch.load(train_target_file_path)
        train_target_tensors.append(train_target_state_dict)

    # Concatenate tensors along the first dimension (assuming they have the same size)
    train_images_tensor = torch.cat(train_images_tensors, dim=0)
    train_target_tensor = torch.cat(train_target_tensors, dim=0)

    #print(train_images_tensor.size())
    
    test_images_file_path = f'/Users/andrewblumensen/Documents/mlops/Day1/data/test_images.pt'
    test_images_state_dict = torch.load(test_images_file_path)

    test_target_file_path = f'/Users/andrewblumensen/Documents/mlops/Day1/data/test_target.pt'
    test_target_state_dict = torch.load(test_target_file_path)

    return train_images_tensor, train_target_tensor
