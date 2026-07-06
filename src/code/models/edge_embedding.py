import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeEmbedding(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int = None):
        super().__init__()
        self.edge_dim = node_dim if edge_dim is None else edge_dim
        self.proj = nn.Linear(node_dim * 2, self.edge_dim)

    def _embed_pairs(self, src_nodes, dst_nodes):
        return F.relu(self.proj(torch.cat([src_nodes, dst_nodes], dim=-1)))

    def forward(self, node_embeddings, edge_index):
        row, col = edge_index
        return self._embed_pairs(node_embeddings[row], node_embeddings[col])

    def self_loops(self, node_embeddings):
        return self._embed_pairs(node_embeddings, node_embeddings)
