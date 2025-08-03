import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(batch_size=64, val_split=0.2, data_dir='./data'):
    """
    Downloads the CIFAR-10 dataset and creates training, validation, and test dataloaders.

    Args:
        batch_size (int): The number of samples per batch.
        val_split (float): The proportion of the training set to use for validation.
        data_dir (str): The directory to store the dataset.

    Returns:
        tuple: A tuple containing (train_loader, val_loader, test_loader).
    """
    
    # --- 1. Define Transformations ---
    
    # Transformations for the training set with data augmentation
    # We use random cropping and horizontal flipping to make the model more robust.
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # Normalization values are standard for CIFAR-10
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Transformations for the validation and test sets
    # No data augmentation is needed here, just normalization.
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # --- 2. Download and Load Datasets ---
    
    # Download the full training dataset
    full_train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=True,
        download=True, 
        transform=train_transform
    )

    # Download the test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, 
        train=False,
        download=True, 
        transform=test_transform
    )

    # --- 3. Create Training and Validation Splits ---
    
    # Calculate the number of samples for training and validation
    num_train = len(full_train_dataset)
    val_size = int(val_split * num_train)
    train_size = num_train - val_size
    
    # Split the full training dataset
    # We use a fixed generator for reproducibility of the split
    train_dataset, val_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42) # for a consistent split
    )
    
    # IMPORTANT: We need to apply the test transformations to the validation set.
    # The 'val_dataset' object is a Subset, so we access its underlying dataset.
    # We clone the dataset to avoid modifying the original train_dataset.
    # This is a common practice to ensure validation data is not augmented.
    val_dataset.dataset = test_dataset.__class__(
        root=data_dir, 
        train=True, 
        download=False, # Already downloaded
        transform=test_transform
    )


    # --- 4. Create DataLoaders ---
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2, # Use multiple subprocesses to load data
        pin_memory=True # For faster data transfer to the GPU
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, # No need to shuffle validation data
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, # No need to shuffle test data
        num_workers=2,
        pin_memory=True
    )

    print(f"Datasets loaded successfully.")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # This block allows you to test the script directly
    print("Testing the data loader script...")
    
    # Get the dataloaders with default settings
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)
    
    # Fetch one batch from the training loader to verify
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    
    print("\n--- Verification ---")
    print(f"Image batch shape: {images.shape}") # Should be [batch_size, 3, 32, 32]
    print(f"Label batch shape: {labels.shape}") # Should be [batch_size]
    print("Data loader test complete.")


