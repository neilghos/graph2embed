import torch
import torch.nn as nn
import torch.nn.functional as F

def random_walk(row, col, start, walk_length, num_nodes):
    """
    Custom vectorized random walk for small/medium graphs.
    Returns: [num_starts, walk_length + 1] tensor of node indices.
    """
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float, device=row.device)
    adj[row, col] = 1.0
    
    walks = [start]
    curr = start
    for _ in range(walk_length):
        weights = adj[curr] # [num_starts, num_nodes]
        # Handle nodes with no outgoing edges by adding a self-loop
        row_sums = weights.sum(dim=1, keepdim=True)
        weights = torch.where(row_sums == 0, torch.ones_like(weights), weights)
        
        # Sample one neighbor per walk
        next_nodes = torch.multinomial(weights, 1).squeeze(1)
        walks.append(next_nodes)
        curr = next_nodes
        
    return torch.stack(walks, dim=1)

class ConvReaderGraphEmbedding(nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, walk_length=5, walks_per_node=5, conv_kernel_size=3):
        super().__init__()
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.hidden_dim = hidden_dim
        
        # 1. Node Embedding
        self.node_emb = nn.Linear(num_node_features, hidden_dim)
        
        # 2. Edge Embedding: Linear(concat(u, v))
        self.edge_emb = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 3. Convolutional Reader
        # The input to Conv1d expects (batch_size, in_channels, sequence_length)
        # Our chain has length: walk_length (nodes) + walk_length (edges) = 2 * walk_length + 1 items?
        # A walk of length L has L+1 nodes and L edges.
        # Alternating chain: N0, E01, N1, E12, N2 ...
        # Total length = (L+1) + L = 2L + 1
        self.conv1d = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=conv_kernel_size, padding=conv_kernel_size//2)
        
        # 4. Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        num_nodes = x.size(0)
        
        # 1. Node Embeddings
        # x is [num_nodes, num_node_features]
        h_nodes = F.relu(self.node_emb(x)) # [num_nodes, hidden_dim]
        
        # 2. Extract Random Walks
        row, col = edge_index
        # start nodes: repeat each node 'walks_per_node' times
        start = torch.arange(num_nodes, device=x.device).repeat(self.walks_per_node)
        
        # walks: [num_starts, walk_length + 1] containing node indices
        walks = random_walk(row, col, start, walk_length=self.walk_length, num_nodes=num_nodes)
        
        num_walks = walks.size(0)
        
        # We need to construct the alternating sequence for each walk
        # walks[:, i] is the i-th node in the walk.
        # Edge from walks[:, i] to walks[:, i+1]
        
        sequence_tensors = []
        for i in range(self.walk_length + 1):
            # Node embedding
            n_idx = walks[:, i]
            # Replace -1 (padding/terminated walks) with a dummy node or handle it.
            # random_walk pads with -1 if it can't continue. 
            # PyG's random_walk doesn't pad with -1 if the graph has no isolated nodes or dead ends usually, but let's clamp.
            n_idx_clamped = n_idx.clamp(min=0)
            n_emb = h_nodes[n_idx_clamped] # [num_walks, hidden_dim]
            
            # Zero out embeddings for padded steps
            mask = (n_idx != -1).float().unsqueeze(-1)
            n_emb = n_emb * mask
            sequence_tensors.append(n_emb)
            
            if i < self.walk_length:
                # Edge embedding between walks[:, i] and walks[:, i+1]
                next_idx = walks[:, i+1]
                next_idx_clamped = next_idx.clamp(min=0)
                next_n_emb = h_nodes[next_idx_clamped]
                
                # Concat u and v
                uv_concat = torch.cat([n_emb, next_n_emb], dim=-1) # [num_walks, 2 * hidden_dim]
                e_emb = F.relu(self.edge_emb(uv_concat)) # [num_walks, hidden_dim]
                
                # Zero out if either node is invalid
                next_mask = (next_idx != -1).float().unsqueeze(-1)
                e_emb = e_emb * mask * next_mask
                sequence_tensors.append(e_emb)
                
        # Stack sequence: [num_walks, seq_len, hidden_dim]
        # seq_len = 2 * walk_length + 1
        seq = torch.stack(sequence_tensors, dim=1) 
        
        # Conv1d expects [batch, channels, seq_len]
        seq = seq.transpose(1, 2) # [num_walks, hidden_dim, seq_len]
        
        # 3. Convolutional Reader
        conv_out = F.relu(self.conv1d(seq)) # [num_walks, hidden_dim, seq_len]
        
        # 4. Pooling over the sequence (path embedding)
        # Max pool over seq_len
        path_emb, _ = torch.max(conv_out, dim=-1) # [num_walks, hidden_dim]
        
        # 5. Graph-level pooling
        # We need to map each walk back to the graph it belongs to.
        # start nodes map to graphs via 'batch'
        walk_batch = batch[start] # [num_walks]
        
        # Scatter max to pool path embeddings into graph embeddings
        # Scatter max to pool path embeddings into graph embeddings natively in PyTorch
        graph_emb = path_emb.new_zeros((data.num_graphs, path_emb.size(1)))
        index = walk_batch.unsqueeze(-1).expand_as(path_emb)
        graph_emb.scatter_reduce_(0, index, path_emb, reduce="amax", include_self=False)
        
        # 6. Classify
        logits = self.classifier(graph_emb)
        
        return logits
