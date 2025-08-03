import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt

# --- Import your custom modules ---
from AlexNet import AlexNet_modified  
from get_data import get_dataloaders 

# ----------------- HYPERPARAMETERS -----------------
BATCH_SIZE = 128
LEARNING_RATE = 0.001
NUM_EPOCHS = 40 # You can increase this for better performance
VAL_SPLIT = 0.15
MODEL_SAVE_PATH = 'best_alexnet_modified.pth'

# ------------------Logging-------------------
if not os.path.exists('logs_alexnet'):
    os.makedirs('logs_alexnet')
    
training_history = {
    'train_loss': [],
    'val_loss': [],
    'train_acc': [],
    'val_acc': [],
    }


def get_device():
    """Checks for and returns the available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        print("Using CUDA (GPU)")
        return torch.device('cuda')
    else:
        print("Using CPU")
        return torch.device('cpu')

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Performs one full training pass over the training data.
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)

        # --- Forward pass ---
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # --- Backward pass and optimization ---
        optimizer.zero_grad() # Clear previous gradients
        loss.backward()       # Compute gradients
        optimizer.step()      # Update weights
        
        # --- Statistics ---
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        # Update progress bar
        progress_bar.set_postfix(loss=loss.item())
        
    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions / total_samples) * 100
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation or test set.
    """
    model.eval() 
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    progress_bar = tqdm(dataloader, desc='Validation', leave=False)
    
    with torch.no_grad(): 
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / total_samples
    epoch_acc = (correct_predictions / total_samples) * 100
    return epoch_loss, epoch_acc

# ----------------- MAIN TRAINING SCRIPT -----------------
if __name__ == '__main__':
    device = get_device()
    
    # --- 1. Load Data ---
    print("Loading CIFAR-10 dataset...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, 
        val_split=VAL_SPLIT
    )

    # --- 2. Initialize Model, Loss, and Optimizer ---
    print("Initializing model...")
    model = AlexNet_modified(num_classes=10).to(device)
    num_params = model.num_parameters()
    print(f"Model has {num_params} parameters.")

    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer (Adam is a good default choice)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. Training Loop ---
    best_val_accuracy = 0.0
    print("\nStarting training...")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1} Summary:")
        print(f"\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"\tValid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.2f}%")

        # Log the training and validation metrics
        with open('logs_alexnet/training_log.txt', 'a') as f:
            f.write(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%,"
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n") 
            
        # Save the training history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_acc'].append(val_acc)     
        
        # Save the model if it has the best validation accuracy so far
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"\tNew best model saved to {MODEL_SAVE_PATH} (Accuracy: {val_acc:.2f}%)")

    end_time = time.time()
    print(f"\nTraining finished in {(end_time - start_time)/60:.2f} minutes.")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")

    # --- 4. Final Evaluation on Test Set ---
    print("\nLoading best model for final testing...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    test_loss, test_acc = validate_one_epoch(model, test_loader, criterion, device)
    print(f"\nFinal Test Results:")
    print(f"\tTest Loss: {test_loss:.4f}")
    print(f"\tTest Accuracy: {test_acc:.2f}%")

    # Log the test results
    with open('logs_alexnet/test_log.txt', 'a') as f:
        f.write(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%\n") 

# Save the final model
torch.save(model.state_dict(), 'final_alexnet_modified.pth')
print("Final model saved as 'final_alexnet_modified.pth'.")

# --- 5. Visualize Training History ---
def plot_training_history(training_history):
    """
    Plots the training and validation loss and accuracy over epochs.
    """
    epochs = range(1, len(training_history['train_loss']) + 1)

    plt.figure(figsize=(12, 4))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_history['train_loss'], label='Train Loss')
    plt.plot(epochs, training_history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, training_history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()
    # Example usage
    training_history = {
        'train_loss': [],  # Collect these during training
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    training_history['train_loss'].append(train_loss)
    training_history['val_loss'].append(val_loss)
    training_history['train_acc'].append(train_acc)
    training_history['val_acc'].append(val_acc)

     
# Call the plotting function to visualize the training history
plot_training_history(training_history)
print("Training history plotted successfully.") 
