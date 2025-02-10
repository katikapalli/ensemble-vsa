import os
import yaml
from torch.utils.data import DataLoader
from src.data.transforms import get_train_transforms, get_valid_transforms, get_test_transforms
from src.data.dataset import AffectNetDataset

def get_data_loaders(root_dir, config, batch_size=32, num_workers=4, image_size=224):
    """
    Args:
        root_dir (string): Root directory containing the dataset.
        config (dict): Configuration dictionary containing information about the dataset.
        batch_size (int): Number of samples per batch (default: 32).
        num_workers (int): Number of workers for data loading (default: 4).
        image_size (int): The size to which the images will be resized (default: 224).
        
    Returns:
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
    """
    train_transforms = get_train_transforms(image_size)
    valid_transforms = get_valid_transforms(image_size)
    test_transforms = get_test_transforms(image_size)

    train_dataset = AffectNetDataset(root_dir=root_dir, transform=train_transforms, config=config, split='train')
    valid_dataset = AffectNetDataset(root_dir=root_dir, transform=valid_transforms, config=config, split='valid')
    test_dataset = AffectNetDataset(root_dir=root_dir, transform=test_transforms, config=config, split='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

    return train_loader, valid_loader, test_loader