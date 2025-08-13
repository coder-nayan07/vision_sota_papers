import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data_loader import get_dataloaders
from model import GoogLeNet

# --- Configuration ---
DATA_DIR = '/raid/home/dgx758/shailesh/nayan/vision/tiny-imagenet-200'
MODEL_TYPE = 'GoogLeNet'
BATCH_SIZE = 256 # Adjust based on your GPU memory
EPOCHS = 50       
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- Setup Run Directory ---
RUN_ID = f'run_{MODEL_TYPE}_{time.strftime("%Y%m%d_%H%M%S")}'
os.makedirs(RUN_ID, exist_ok=True)

# --- File Paths for Outputs ---
LOG_FILE = os.path.join(RUN_ID, 'training.log')
HISTORY_FILE = os.path.join(RUN_ID, 'history.json')
PLOT_FILE = os.path.join(RUN_ID, 'accuracy_plot.png')
MODEL_SAVE_FILE = os.path.join(RUN_ID, 'best_model.pth')

def log_message(message, file_handle):
    """Prints a message and writes it to the log file."""
    print(message)
    file_handle.write(message + '\n')

def calculate_accuracy(outputs, labels):
    """Calculates Top-1 and Top-5 accuracy."""
    # Top-1 accuracy
    _, pred_top1 = torch.max(outputs, 1)
    correct_top1 = (pred_top1 == labels).sum().item()
    
    # Top-5 accuracy
    _, pred_top5 = torch.topk(outputs, 5, dim=1)
    labels_reshaped = labels.view(-1, 1).expand_as(pred_top5)
    correct_top5 = (pred_top5 == labels_reshaped).sum().item()
    
    return correct_top1, correct_top5

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    total_correct_top1, total_correct_top5, total_samples = 0, 0, 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # GoogLeNet returns three outputs in train mode
        main_out, aux1_out, aux2_out = model(inputs)
        
        loss1 = criterion(main_out, labels)
        loss2 = criterion(aux1_out, labels)
        loss3 = criterion(aux2_out, labels)
        
        # Weighted loss as per the paper
        loss = loss1 + 0.3 * (loss2 + loss3)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Accuracy is calculated on the main output
        correct_top1, correct_top5 = calculate_accuracy(main_out, labels)
        total_correct_top1 += correct_top1
        total_correct_top5 += correct_top5
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_top1_acc = (total_correct_top1 / total_samples) * 100
    epoch_top5_acc = (total_correct_top5 / total_samples) * 100
    
    return epoch_loss, epoch_top1_acc, epoch_top5_acc

def evaluate(model, dataloader, criterion, device):
    """Evaluates the model on the validation set."""
    model.eval()
    running_loss = 0.0
    total_correct_top1, total_correct_top5, total_samples = 0, 0, 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # GoogLeNet returns one output in eval mode
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            correct_top1, correct_top5 = calculate_accuracy(outputs, labels)
            total_correct_top1 += correct_top1
            total_correct_top5 += correct_top5
            total_samples += labels.size(0)
            
    epoch_loss = running_loss / total_samples
    epoch_top1_acc = (total_correct_top1 / total_samples) * 100
    epoch_top5_acc = (total_correct_top5 / total_samples) * 100
    
    return epoch_loss, epoch_top1_acc, epoch_top5_acc

def save_plot(history, epochs, file_path):
    """Saves a plot of training and validation accuracy."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    epochs_range = range(1, epochs + 1)
    ax.plot(epochs_range, history['train_top1_acc'], 'o-', label='Training Top-1 Accuracy')
    ax.plot(epochs_range, history['val_top1_acc'], 'o-', label='Validation Top-1 Accuracy')
    ax.plot(epochs_range, history['train_top5_acc'], 's--', label='Training Top-5 Accuracy')
    ax.plot(epochs_range, history['val_top5_acc'], 's--', label='Validation Top-5 Accuracy')
    
    ax.set_title(f'Training and Validation Accuracy ({MODEL_TYPE})', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True)
    
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    history = {
        'train_loss': [], 'train_top1_acc': [], 'train_top5_acc': [],
        'val_loss': [], 'val_top1_acc': [], 'val_top5_acc': []
    }
    best_val_acc = 0.0

    with open(LOG_FILE, 'w') as log_f:
        log_message(f"Starting New Run: {RUN_ID}", log_f)
        log_message(f"Device: {DEVICE}", log_f)
        
        # --- Data Loading ---
        log_message("Loading Tiny ImageNet data...", log_f)
        train_loader, val_loader, num_classes = get_dataloaders(DATA_DIR, BATCH_SIZE)
        
        # --- Model Initialization ---
        log_message(f"Initializing {MODEL_TYPE} model with {num_classes} classes...", log_f)
        model = GoogLeNet(num_classes=num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        mode='min',     # The scheduler will look at the validation loss
                                                        factor=0.2,     # Factor by which the learning rate will be reduced. new_lr = lr * factor
                                                        patience=3,     # Number of epochs with no improvement after which learning rate will be reduced
                                                        verbose=True)   # Prints a message when the learning rate is updated

        # --- Training Loop ---
        log_message(f"Starting training for {EPOCHS} epochs...", log_f)
        start_time = time.time()
        
        for epoch in range(EPOCHS):
            epoch_start_time = time.time()
            
            train_loss, train_top1, train_top5 = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_top1, val_top5 = evaluate(model, val_loader, criterion, DEVICE)
            
            scheduler.step(val_loss)
            
            # Store history
            history['train_loss'].append(train_loss)
            history['train_top1_acc'].append(train_top1)
            history['train_top5_acc'].append(train_top5)
            history['val_loss'].append(val_loss)
            history['val_top1_acc'].append(val_top1)
            history['val_top5_acc'].append(val_top5)
            
            epoch_duration = time.time() - epoch_start_time
            
            log_message(
                f"Epoch {epoch+1:02}/{EPOCHS} | "
                f"Time: {epoch_duration:.2f}s | "
                f"LR: {optimizer.param_groups[0]['lr']:.5f} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}",
                log_f
            )
            log_message(
                f"  -> Train Acc: (Top1: {train_top1:.2f}%, Top5: {train_top5:.2f}%) | "
                f"Val Acc: (Top1: {val_top1:.2f}%, Top5: {val_top5:.2f}%)",
                log_f
            )
            

            if val_top1 > best_val_acc:
                best_val_acc = val_top1
                torch.save(model.state_dict(), MODEL_SAVE_FILE)
                log_message(f"  -> New best model saved with Val Acc: {best_val_acc:.2f}%", log_f)
        
        total_duration = time.time() - start_time
        log_message(f"\nTraining finished in {total_duration / 60:.0f}m {total_duration % 60:.0f}s", log_f)

        # --- Save Final Results ---
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)
        log_message(f"Training history saved to {HISTORY_FILE}", log_f)
        
        save_plot(history, EPOCHS, PLOT_FILE)
        log_message(f"Accuracy plot saved to {PLOT_FILE}", log_f)

if __name__ == '__main__':
    main()