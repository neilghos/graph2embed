import ssl
ssl.create_default_context = ssl._create_unverified_context

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch

def get_mutag_dataset(batch_size=32):
    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    
    # Shuffle and split
    dataset = dataset.shuffle()
    
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset = dataset[:train_size]
    val_dataset = dataset[train_size:train_size + val_size]
    test_dataset = dataset[train_size + val_size:]
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return dataset, train_loader, val_loader, test_loader
