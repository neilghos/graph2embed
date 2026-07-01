import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch_geometric.nn import GINConv, global_add_pool
from dataset import get_dataset

class StandardGIN(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, num_layers=5):
        super().__init__()
        
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(GINConv(
            nn.Sequential(
                nn.Linear(num_node_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        ))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            ))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
        # Linear classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GIN layers
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))
            
        # Global Readout
        x = global_add_pool(x, batch)
        
        # Classify
        return self.classifier(x)

def train_baseline(dataset_name):
    print(f"Loading dataset: {dataset_name} for Standard GIN Baseline...")
    dataset, train_loader, val_loader, test_loader = get_dataset(name=dataset_name, batch_size=32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    num_node_features = dataset.num_node_features
    num_classes = dataset.num_classes
    
    model = StandardGIN(
        num_node_features=max(1, num_node_features), # fallback for 0 feature datasets
        hidden_dim=64,
        num_classes=num_classes,
        num_layers=5
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # Standard GIN LR is often 0.01
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
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            
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
    print(f"\nBaseline GIN Final Test Accuracy: {test_acc:.4f}")
    print(f"Baseline GIN Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Standard GIN Baseline")
    parser.add_argument('--dataset', type=str, default='MUTAG', help='TUDataset name (e.g. MUTAG, PROTEINS, NCI1)')
    args = parser.parse_args()
    
    train_baseline(args.dataset)
