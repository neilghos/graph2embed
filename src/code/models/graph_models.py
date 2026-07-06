import torch
import numpy as np
import scipy as sp
import torch.nn.functional as F
import pytorch_lightning as pl
import torch_geometric

from collections import defaultdict
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, matthews_corrcoef
from torch.nn import Linear, BatchNorm1d, ReLU, Dropout
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, global_add_pool, global_mean_pool, global_max_pool


def get_regression_metrics(y_true, y_pred):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    errors = y_true - y_pred
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(np.power(errors, 2)))
    maxer = np.max(np.abs(errors))
    r2, pval = np.power(sp.stats.pearsonr(y_true.flatten(), y_pred.flatten()), 2)

    return (mae, rmse, maxer, r2, np.sqrt(pval))


def get_classification_metrics(y_true, y_pred, digits=6):
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    # ROC AUC not defined if a single label is present
    except ValueError:
        roc_auc = None

    return confusion_matrix(y_true, y_pred), roc_auc, classification_report(y_true, y_pred, digits=digits), matthews_corrcoef(y_true, y_pred)

class GNN(pl.LightningModule):
    def __init__(self,
                 conv_type: str,
                 in_channels: int,
                 gnn_intermediate_dim: int,
                 gnn_output_node_dim: int,
                 output_nn_intermediate_dim: int,
                 output_nn_out_dim: int,
                 task_type: str,
                 readout: str,
                 loss_metric: str,
                 learning_rate: float,
                 num_layers: int,
                 gat_heads: int = None,
                 gat_dropouts: int = None,
                 walk_length: int = 5,
                 walks_per_node: int = 5,
                 ):
        super(GNN, self).__init__()

        self.conv_type = conv_type
        self.in_channels = in_channels
        self.gnn_intermediate_dim = gnn_intermediate_dim
        self.gnn_output_node_dim = gnn_output_node_dim
        self.output_nn_intermediate_dim = output_nn_intermediate_dim
        self.output_nn_out_dim = output_nn_out_dim
        self.task_type = task_type
        self.readout = readout
        self.loss_metric = loss_metric
        self.learning_rate = learning_rate

        self.gat_heads = gat_heads
        self.gat_dropouts = gat_dropouts

        self.walk_length = walk_length
        self.walks_per_node = walks_per_node
        self.num_layers = num_layers

        # Standard readouts preserve the node dimension; our walk reader is bidirectional.
        if self.readout == 'ours':
            self.graph_dim = self.gnn_output_node_dim * 2
        else:
            self.graph_dim = self.gnn_output_node_dim

        # Storage
        self.train_outputs = defaultdict(list)
        self.validation_outputs = defaultdict(list)
        self.test_outputs = defaultdict(list)

        self.train_metrics_per_epoch = {}
        self.validation_metrics_per_epoch = {}
        self.test_metrics_per_epoch = {}
        self.test_graphs_per_epoch = {}

        # Input assertions
        assert self.conv_type in ['GCN', 'GATv2', 'GIN']
        assert self.task_type in ['regression', 'binary_classification', 'multi_classification']
        assert self.readout in ['sum', 'mean', 'max', 'ours']
        assert self.loss_metric in ['MAE', 'MSE', 'BCEWithLogits', 'CrossEntropyLoss']

        print(f'Training with {self.num_layers} layers.')

        # Convolutional layers
        convs = []

        # GCN
        if self.conv_type == 'GCN':
            for i in range(self.num_layers):
                if i == 0:
                    convs.append((GCNConv(in_channels=self.in_channels, out_channels=self.gnn_intermediate_dim, cached=False,  normalize=True), 'x, edge_index -> x'))
                elif i != self.num_layers - 1:
                    convs.append((GCNConv(in_channels=self.gnn_intermediate_dim, out_channels=self.gnn_intermediate_dim, cached=False, normalize=True), 'x, edge_index -> x'))
                else:
                    convs.append((GCNConv(in_channels=self.gnn_intermediate_dim, out_channels=self.gnn_output_node_dim, cached=False, normalize=True), 'x, edge_index -> x'))
                convs.append(ReLU(inplace=True))

        # GATv2
        if self.conv_type == 'GATv2':
            for i in range(self.num_layers):
                if i == 0:
                    convs.append((GATv2Conv(in_channels=self.in_channels, out_channels=self.gnn_intermediate_dim, heads=self.gat_heads,
                                            concat=True, dropout=self.gat_dropouts), 'x, edge_index -> x'))
                elif i != self.num_layers - 1:
                    convs.append((GATv2Conv(in_channels=self.gnn_intermediate_dim * self.gat_heads, out_channels=self.gnn_intermediate_dim,
                                            heads=self.gat_heads, concat=True, dropout=self.gat_dropouts), 'x, edge_index -> x'))
                else:
                    convs.append((GATv2Conv(in_channels=self.gnn_intermediate_dim * self.gat_heads, out_channels=self.gnn_output_node_dim,
                                            heads=self.gat_heads, concat=False, dropout=self.gat_dropouts), 'x, edge_index -> x'))
                convs.append(ReLU(inplace=True))

        # GIN
        if self.conv_type == 'GIN':
            for i in range(self.num_layers):
                if i == 0:
                    convs.append((GINConv(
                            torch.nn.Sequential(Linear(in_features=self.in_channels, out_features=self.gnn_intermediate_dim),
                            BatchNorm1d(self.gnn_intermediate_dim),
                            ReLU(),
                            Linear(in_features=self.gnn_intermediate_dim, out_features=self.gnn_intermediate_dim),
                            ReLU()
                           )
                        ), 'x, edge_index -> x'))
                elif i != self.num_layers - 1:
                    convs.append((GINConv(
                            torch.nn.Sequential(Linear(in_features=self.gnn_intermediate_dim, out_features=self.gnn_intermediate_dim),
                            BatchNorm1d(self.gnn_intermediate_dim),
                            ReLU(),
                            Linear(in_features=self.gnn_intermediate_dim, out_features=self.gnn_intermediate_dim),
                            ReLU()
                            )
                        ), 'x, edge_index -> x'))
                else:
                    convs.append((GINConv(
                            torch.nn.Sequential(Linear(in_features=self.gnn_intermediate_dim, out_features=self.gnn_output_node_dim),
                            BatchNorm1d(self.gnn_output_node_dim),
                            ReLU(),
                            Linear(in_features=self.gnn_output_node_dim, out_features=self.gnn_output_node_dim),
                            ReLU()
                            )
                        ), 'x, edge_index -> x'))
                convs.append(ReLU(inplace=True))

        self.convs = torch_geometric.nn.Sequential('x, edge_index', convs)


        if self.readout == 'ours':
            from models.edge_embedding import EdgeEmbedding
            from models.lstmreader import LSTMReader
            self.edge_embedder = EdgeEmbedding(node_dim=self.gnn_output_node_dim)
            self.lstm_reader = LSTMReader(
                node_dim=self.gnn_output_node_dim,
                walk_length=self.walk_length,
                walks_per_node=self.walks_per_node,
            )


        # Regression/classification NN
        in_dim = self.graph_dim

        self.output_nn = torch.nn.Sequential(
            Linear(in_features=in_dim, out_features=self.output_nn_intermediate_dim),
            ReLU(),
            Dropout(p=0.2),
            Linear(in_features=self.output_nn_intermediate_dim, out_features=self.output_nn_out_dim)
        )


    def forward(self, x, edge_index, batch):
        x = x.float()

        x = self.convs(x, edge_index)

        if self.readout == 'sum':
            graph_x = global_add_pool(x, batch)

        elif self.readout == 'mean':
            graph_x = global_mean_pool(x, batch)

        elif self.readout == 'max':
            graph_x = global_max_pool(x, batch)

        elif self.readout == 'ours':
            edge_embeddings = self.edge_embedder(x, edge_index)
            self_loop_edge_embeddings = self.edge_embedder.self_loops(x)
            graph_x = self.lstm_reader(
                node_embeddings=x,
                edge_index=edge_index,
                edge_embeddings=edge_embeddings,
                self_loop_edge_embeddings=self_loop_edge_embeddings,
                batch=batch,
            )

        task_predictions = self.output_nn(graph_x)

        return task_predictions, graph_x


    def task_loss(self, y_pred, y_true):
        if self.loss_metric == 'BCEWithLogits':
            y_true = y_true.view(y_pred.shape)
            # All labels are 0/1, so binary classification. There might be 1 or more labels.
            task_loss = F.binary_cross_entropy_with_logits(y_pred.float(), y_true.float())

        elif self.loss_metric == 'CrossEntropyLoss':
            task_loss = F.cross_entropy(y_pred.float(), y_true.long())

        elif self.loss_metric == 'MSE':
            y_true = y_true.view(y_pred.shape)
            task_loss = F.mse_loss(y_pred, y_true.float())

        elif self.loss_metric == 'MAE':
            y_true = y_true.view(y_pred.shape)
            task_loss = F.l1_loss(y_pred, y_true.float())

        return task_loss


    def _step(self, batch, batch_idx):
        x, edge_index, y, batch_ids = batch.x, batch.edge_index, batch.y, batch.batch
        task_predictions, graph_x = self.forward(x=x, edge_index=edge_index, batch=batch_ids)

        loss = self.task_loss(task_predictions, y)

        return loss, y, task_predictions, graph_x


    def training_step(self, batch, batch_idx):
        loss, ys, task_predictions, graph_x = self._step(batch, batch_idx)

        self.log('train_total_loss', loss)
        self.train_outputs[self.current_epoch].append({'y_true': ys, 'y_pred': task_predictions})

        return loss


    def validation_step(self, batch, batch_idx):
        loss, ys, task_predictions, graph_x = self._step(batch, batch_idx)

        self.log('validation_total_loss', loss)
        self.validation_outputs[self.current_epoch].append({'y_true': ys, 'y_pred': task_predictions})

        return loss


    def test_step(self, batch, batch_idx):
        loss, ys, task_predictions, graph_x = self._step(batch, batch_idx)

        self.test_outputs[self.current_epoch].append({'y_true': ys, 'y_pred': task_predictions, 'graph_x': graph_x})

        return loss


    def _get_metrics_epoch_end(self, all_y_true, all_y_pred):
        if self.task_type == 'regression':
            all_y_true = all_y_true.detach().cpu().numpy()
            all_y_pred = all_y_pred.detach().cpu().numpy()

            if all_y_true.shape != all_y_pred.shape:
                all_y_pred = all_y_pred.reshape(all_y_true.shape)

            metrics = get_regression_metrics(y_true=all_y_true, y_pred=all_y_pred)
            return metrics

        elif self.task_type == 'binary_classification':
            all_y_pred = torch.sigmoid(all_y_pred)
            all_y_pred = torch.where(all_y_pred >= 0.5, 1.0, 0.0).long()

        elif self.task_type == 'multi_classification':
            all_y_pred_softmax = torch.log_softmax(all_y_pred, dim = 1)
            _, all_y_pred = torch.max(all_y_pred_softmax, dim = 1)

        if 'classification' in self.task_type:
            all_y_pred = all_y_pred.view(all_y_true.shape).squeeze()

        return get_classification_metrics(y_true=all_y_true.long().detach().cpu().numpy(), y_pred=all_y_pred.detach().cpu().numpy())


    def _log_epoch_metrics(self, prefix, metrics):
        if self.task_type == 'regression':
            mae, _, _, r2, _ = metrics
            self.log(f'{prefix}_mae', float(mae), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{prefix}_r2', float(r2), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        else:
            _, roc_auc, _, mcc = metrics
            if roc_auc is not None:
                self.log(f'{prefix}_auroc', float(roc_auc), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f'{prefix}_mcc', float(mcc), on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def on_train_epoch_end(self, unused=None):
        all_y_true = [elem['y_true'] for elem in self.train_outputs[self.current_epoch]]
        all_y_pred = [elem['y_pred'] for elem in self.train_outputs[self.current_epoch]]

        all_y_true = torch.cat(all_y_true, dim=0)
        all_y_pred = torch.cat(all_y_pred, dim=0)

        metrics = self._get_metrics_epoch_end(all_y_true, all_y_pred)

        self.train_metrics_per_epoch[self.current_epoch] = metrics
        self._log_epoch_metrics('train', metrics)

        del self.train_outputs[self.current_epoch]
        del all_y_true
        del all_y_pred


    def on_validation_epoch_end(self, unused=None):
        all_y_true = [elem['y_true'] for elem in self.validation_outputs[self.current_epoch]]
        all_y_pred = [elem['y_pred'] for elem in self.validation_outputs[self.current_epoch]]

        all_y_true = torch.cat(all_y_true, dim=0)
        all_y_pred = torch.cat(all_y_pred, dim=0)

        metrics = self._get_metrics_epoch_end(all_y_true, all_y_pred)

        self.validation_metrics_per_epoch[self.current_epoch] = metrics
        self._log_epoch_metrics('validation', metrics)

        del self.validation_outputs[self.current_epoch]
        del all_y_true
        del all_y_pred


    def on_test_epoch_end(self, unused=None):
        all_y_true = [elem['y_true'] for elem in self.test_outputs[self.current_epoch]]
        all_y_pred = [elem['y_pred'] for elem in self.test_outputs[self.current_epoch]]
        all_graph_x = [elem['graph_x'] for elem in self.test_outputs[self.current_epoch]]

        all_y_true = torch.cat(all_y_true, dim=0)
        all_y_pred = torch.cat(all_y_pred, dim=0)
        all_graph_x = torch.cat(all_graph_x, dim=0)

        metrics = self._get_metrics_epoch_end(all_y_true, all_y_pred)

        self.test_metrics_per_epoch[self.current_epoch] = metrics
        self.test_graphs_per_epoch[self.current_epoch] = all_graph_x.detach().cpu().numpy()
        self._log_epoch_metrics('test', metrics)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('GNN')
        parser.add_argument('--conv_type', type=str)
        parser.add_argument('--gnn_intermediate_dim', type=int)
        parser.add_argument('--gnn_output_node_dim', type=int)
        parser.add_argument('--output_nn_intermediate_dim', type=int)
        parser.add_argument('--readout', type=str)
        parser.add_argument('--learning_rate', type=float)

        parser.add_argument('--gat_heads', type=int)
        parser.add_argument('--gat_dropouts', type=float)
        parser.add_argument('--walk_length', type=int, default=5)
        parser.add_argument('--walks_per_node', type=int, default=5)

        return parent_parser
