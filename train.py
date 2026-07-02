import torch
import torch.nn as nn
import argparse
import random
import numpy as np
from dataset import get_dataset
from model import LSTMReaderGraphEmbedding

import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # Enforce deterministic algorithms, fallback to warning if not available
        torch.use_deterministic_algorithms(True, warn_only=True)

def train(dataset_name, seed=42):
    set_seed(seed) # Fix seed for reproducible splits, initialization, and random walks
    
    print(f"Loading dataset: {dataset_name} (Seed: {seed})...")
    dataset, train_loader, val_loader, test_loader = get_dataset(name=dataset_name, batch_size=32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    num_node_features = dataset.num_node_features
    # If the dataset has no node features (e.g., IMDB-BINARY), we typically use a constant feature of 1.
    # PyG doesn't automatically do this for all datasets, but GIN can work with one-hot degree or constants.
    # For now, if num_node_features is 0, we'll set it to 1 and handle it in the model (requires small tweak)
    # Actually, many TUDatasets have node labels. Let's stick with standard for now.
    
    num_classes = dataset.num_classes
    
    model = LSTMReaderGraphEmbedding(
        num_node_features=max(1, num_node_features), # fallback for 0 feature datasets
        hidden_dim=64,
        num_classes=num_classes,
        walk_length=5,
        walks_per_node=5
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    
    for epoch in range(1, 201):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data in train_loader:
            data = data.to(device)
            # Handle featureless graphs
            if data.x is None:
                data.x = torch.ones((data.num_nodes, 1), device=device)
                
            optimizer.zero_grad()
            
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * data.num_graphs
            pred = out.argmax(dim=1)
            correct += int((pred == data.y).sum())
            total += data.num_graphs
            
        train_acc = correct / total
        train_loss = total_loss / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                if data.x is None:
                    data.x = torch.ones((data.num_nodes, 1), device=device)
                out = model(data)
                pred = out.argmax(dim=1)
                val_correct += int((pred == data.y).sum())
                val_total += data.num_graphs
        
        val_acc = val_correct / val_total
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save the best model
            torch.save(model.state_dict(), f'best_model_{dataset_name}.pth')
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
    # Load the best model before testing
    print(f"\nLoading best model weights for testing...")
    model.load_state_dict(torch.load(f'best_model_{dataset_name}.pth'))
    
    # Test
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            if data.x is None:
                data.x = torch.ones((data.num_nodes, 1), device=device)
            out = model(data)
            pred = out.argmax(dim=1)
            test_correct += int((pred == data.y).sum())
            test_total += data.num_graphs
            
    test_acc = test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    
    return test_acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ConvReader Baseline")
    parser.add_argument('--dataset', type=str, default='MUTAG', help='TUDataset name (e.g. MUTAG, PROTEINS, NCI1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    train(args.dataset, args.seed)
