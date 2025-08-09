
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import matplotlib.pyplot as plt

from data_loader import get_dataloaders
from model import VGGNet

# --- Configuration ---
DATA_DIR = 'tiny-imagenet-200'
MODEL_TYPE = 'VGG16'
BATCH_SIZE = 256
EPOCHS =  60
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- File Paths for Outputs ---
LOG_FILE = 'training_results.log'
HISTORY_FILE = 'training_history.json'
PLOT_FILE = 'accuracy_vs_epoch.jpg'


def log_message(message, file_handle):
    print(message)
    file_handle.write(message + '\n')


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Top-1 accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct_top1 += (predicted == labels).sum().item()
        
        # Top-5 accuracy
        _, top5_pred = torch.topk(outputs.data, 5, dim=1)
        labels_reshaped = labels.view(-1, 1).expand_as(top5_pred)
        correct_top5 += (top5_pred == labels_reshaped).sum().item()
        
        total_samples += labels.size(0)

    epoch_loss = running_loss / total_samples
    epoch_top1_acc = (correct_top1 / total_samples) * 100
    epoch_top5_acc = (correct_top5 / total_samples) * 100
    return epoch_loss, epoch_top1_acc, epoch_top5_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            # Top-1 accuracy
            _, predicted = torch.max(outputs.data, 1)
            correct_top1 += (predicted == labels).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = torch.topk(outputs.data, 5, dim=1)
            labels_reshaped = labels.view(-1, 1).expand_as(top5_pred)
            correct_top5 += (top5_pred == labels_reshaped).sum().item()
            
            total_samples += labels.size(0)
            
    epoch_loss = running_loss / total_samples
    epoch_top1_acc = (correct_top1 / total_samples) * 100
    epoch_top5_acc = (correct_top5 / total_samples) * 100
    return epoch_loss, epoch_top1_acc, epoch_top5_acc


def save_plot(history, epochs, file_path):
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    epochs_range = range(1, epochs + 1)
    ax.plot(epochs_range, history['train_acc'], 'o-', label='Training Top-1 Accuracy')
    ax.plot(epochs_range, history['val_acc'], 'o-', label='Validation Top-1 Accuracy')
    ax.plot(epochs_range, history['train_top5_acc'], 's--', label='Training Top-5 Accuracy')
    ax.plot(epochs_range, history['val_top5_acc'], 's--', label='Validation Top-5 Accuracy')
    
    ax.set_title(f'Training and Validation Accuracy ({MODEL_TYPE})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(loc='lower right')
    ax.grid(True)
    
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    history = {
        'train_loss': [], 'train_acc': [], 'train_top5_acc': [],
        'val_loss': [], 'val_acc': [], 'val_top5_acc': []
    }

    with open(LOG_FILE, 'w') as log_f:
        log_message(f"Using device: {DEVICE}", log_f)
        log_message("Loading data...", log_f)
        
        train_loader, val_loader, num_classes = get_dataloaders(DATA_DIR, BATCH_SIZE)
        
        log_message(f"Initializing {MODEL_TYPE} model with {num_classes} classes...", log_f)
        model = VGGNet(vgg_name=MODEL_TYPE, num_classes=num_classes).to(DEVICE)
        model.count_parameters()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-6)

        log_message(f"Starting training for {EPOCHS} epochs...", log_f)
        start_time = time.time()
        
        for epoch in range(EPOCHS):
            epoch_start_time = time.time()
            
            train_loss, train_acc, train_top5_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            val_loss, val_acc, val_top5_acc = evaluate(model, val_loader, criterion, DEVICE)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_top5_acc'].append(train_top5_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_top5_acc'].append(val_top5_acc)
            
            epoch_duration = time.time() - epoch_start_time
            
            val_top5_error = 100.0 - val_top5_acc
            
            log_message(
                f"Epoch {epoch+1}/{EPOCHS} | "
                f"Time: {epoch_duration:.2f}s | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                f"Val Acc: {val_acc:.2f}% | Val Top-5 Error: {val_top5_error:.2f}%",
                log_f
            )
        
        total_duration = time.time() - start_time
        log_message(f"\nTraining finished in {total_duration / 60:.0f}m {total_duration % 60:.0f}s", log_f)

        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=4)
        log_message(f"Training history saved to {HISTORY_FILE}", log_f)
        
        save_plot(history, EPOCHS, PLOT_FILE)
        log_message(f"Accuracy plot saved to {PLOT_FILE}", log_f)


if __name__ == '__main__':
    main()

