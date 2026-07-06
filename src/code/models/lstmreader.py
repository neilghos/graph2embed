import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool


class LSTMReader(nn.Module):
    def __init__(self, node_dim: int, walk_length: int = 5, walks_per_node: int = 5):
        super().__init__()
        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.node_dim = node_dim
        self.lstm = nn.LSTM(
            input_size=node_dim,
            hidden_size=node_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def _build_walk_sequence(self, node_embeddings, edge_index, edge_embeddings, self_loop_edge_embeddings):
        row, col = edge_index
        perm = torch.argsort(row)
        row = row[perm]
        col = col[perm]
        edge_embeddings = edge_embeddings[perm]

        num_nodes = node_embeddings.size(0)
        counts = torch.bincount(row, minlength=num_nodes)
        ptr = torch.cat([counts.new_zeros(1), counts.cumsum(dim=0)], dim=0)

        start = torch.arange(num_nodes, device=node_embeddings.device).repeat(self.walks_per_node)
        curr = start
        sequence_tensors = []

        for step in range(self.walk_length + 1):
            sequence_tensors.append(node_embeddings[curr])

            if step == self.walk_length:
                break

            deg = counts[curr]
            has_edge = deg > 0
            safe_deg = torch.clamp_min(deg, 1)
            offsets = torch.floor(torch.rand(curr.size(0), device=curr.device) * safe_deg.float()).long()
            edge_positions = ptr[curr] + offsets

            next_nodes = curr.clone()
            step_edge_embeddings = self_loop_edge_embeddings[curr].clone()

            next_nodes[has_edge] = col[edge_positions[has_edge]]
            step_edge_embeddings[has_edge] = edge_embeddings[edge_positions[has_edge]]

            sequence_tensors.append(step_edge_embeddings)
            curr = next_nodes

        return torch.stack(sequence_tensors, dim=1), start

    def forward(self, node_embeddings, edge_index, edge_embeddings, self_loop_edge_embeddings, batch):
        walk_sequence, start = self._build_walk_sequence(
            node_embeddings=node_embeddings,
            edge_index=edge_index,
            edge_embeddings=edge_embeddings,
            self_loop_edge_embeddings=self_loop_edge_embeddings,
        )
        lstm_out, _ = self.lstm(walk_sequence)
        path_embeddings, _ = torch.max(lstm_out, dim=1)
        walk_batch = batch[start]
        return global_max_pool(path_embeddings, walk_batch)
