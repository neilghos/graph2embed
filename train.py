import torch
import torch.nn as nn
from dataset import get_mutag_dataset
from model import ConvReaderGraphEmbedding

def train():
    dataset, train_loader, val_loader, test_loader = get_mutag_dataset(batch_size=32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    num_node_features = dataset.num_node_features
    num_classes = dataset.num_classes
    
    model = ConvReaderGraphEmbedding(
        num_node_features=num_node_features,
        hidden_dim=64,
        num_classes=num_classes,
        walk_length=5,
        walks_per_node=5,
        conv_kernel_size=3
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    
    for epoch in range(1, 201):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for data in train_loader:
            data = data.to(device)
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
            out = model(data)
            pred = out.argmax(dim=1)
            test_correct += int((pred == data.y).sum())
            test_total += data.num_graphs
            
    test_acc = test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")

if __name__ == '__main__':
    train()
