import os
import shutil
import torch
import pandas as pd
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def prepare_validation_data(data_dir, val_dir_name='val'):
    # Rearranges Tiny ImageNet validation data into ImageFolder-compatible subdirectories.
    val_dir = os.path.join(data_dir, val_dir_name)
    val_img_dir = os.path.join(val_dir, 'images')
    
    if not os.path.exists(val_img_dir):
        return

    annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    annotations = pd.read_csv(annotations_file, sep='\t', header=None, names=['file', 'wnid', 'x0', 'y0', 'x1', 'y1'])

    for index, row in annotations.iterrows():
        img_file, wnid = row['file'], row['wnid']
        class_dir = os.path.join(val_dir, wnid)
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        
        src_path = os.path.join(val_img_dir, img_file)
        dest_path = os.path.join(class_dir, img_file)
        shutil.move(src_path, dest_path)

    os.rmdir(val_img_dir)
    print("Validation data processed.")


def get_dataloaders(data_dir, batch_size=128):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    prepare_validation_data(data_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Add color jitter
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Add random shifts
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageFolder(train_dir, transform=train_transform)
    val_dataset = ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
    )
    
    num_classes = len(train_dataset.classes)
    print(f"Found {num_classes} classes.")

    return train_loader, val_loader, num_classes
