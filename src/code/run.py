import os
import re
import ssl
import csv
from importlib.metadata import PackageNotFoundError, version

import numpy as np


def ensure_supported_runtime():
    try:
        numpy_version = version("numpy")
    except PackageNotFoundError:
        return

    numpy_major = int(numpy_version.split(".", maxsplit=1)[0])
    if numpy_major >= 2:
        raise RuntimeError(
            f"Detected NumPy {numpy_version}. This project's Lightning/TorchMetrics stack expects NumPy 1.x. "
            "Downgrade the active environment with `pip install \"numpy<2\"`, or recreate the environment "
            "with the pinned package versions used for this repo."
        )


def configure_ssl_cert_bundle():
    try:
        import certifi
    except ModuleNotFoundError:
        return

    cafile = certifi.where()
    os.environ["SSL_CERT_FILE"] = cafile
    os.environ["REQUESTS_CA_BUNDLE"] = cafile
    os.environ["CURL_CA_BUNDLE"] = cafile

    default_create_context = ssl.create_default_context

    def _certifi_default_https_context(*args, **kwargs):
        kwargs.setdefault("cafile", cafile)
        return default_create_context(*args, **kwargs)

    ssl.create_default_context = _certifi_default_https_context
    ssl._create_default_https_context = _certifi_default_https_context


ensure_supported_runtime()
configure_ssl_cert_bundle()

import torch
import pytorch_lightning as pl
import torch_geometric.transforms as T

from argparse import ArgumentParser
from pathlib import Path
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from pytorch_lightning.callbacks import EarlyStopping
from tqdm.auto import tqdm

from models.graph_models import GNN


MOLNET_DS = ['QM9', 'FreeSolv', 'Lipo', 'HIV', 'BBBP', 'SIDER']
PYG_DS = ['ENZYMES', 'github_stargazers', 'reddit_threads', 'SYNTHETIC', 'SYNTHETICnew',
          'Synthie', 'Cuneiform', 'IMDB-BINARY', 'MUTAG', 'Mutagenicity']
PYG_OTHER_DS = ['ZINC', 'GNNBenchmark_MNIST']
RESULTS_ROOT = Path(__file__).resolve().parents[1] / 'results'

RESULT_FIELDNAMES = [
    'dataset',
    'task_type',
    'row_type',
    'itr',
    'test_epoch',
    'conv_type',
    'readout',
    'num_layers',
    'gnn_intermediate_dim',
    'gnn_output_node_dim',
    'output_nn_intermediate_dim',
    'learning_rate',
    'gat_heads',
    'gat_dropouts',
    'walk_length',
    'walks_per_node',
    'mae',
    'r2',
    'auroc',
    'mcc',
]


def filter_model_only_args(all_argsdict, include_in_channels=True):
    model_keys = ['conv_type', 'gnn_intermediate_dim', 'gnn_output_node_dim', 'output_nn_intermediate_dim', 'readout',
                  'learning_rate', 'gat_heads', 'gat_dropouts',
                  'num_layers', 'walk_length', 'walks_per_node']
    if include_in_channels:
        model_keys.append('in_channels')

    return {k: all_argsdict[k] for k in all_argsdict.keys() if k in model_keys}


def deepchem_iterable_dataset_to_tensors(iter_dataset, use_cuda=False):
    aslist = list(iter_dataset)
    print('Processing DeepChem dataset for PyTorch...')

    new_dataset_as_list = []
    for batch in tqdm(aslist):
        graphs, ys, ws, smiles = batch
        ys = torch.from_numpy(ys.squeeze()).cuda() if use_cuda else torch.from_numpy(ys.squeeze())
        ws = torch.from_numpy(ws.squeeze()).cuda() if use_cuda else torch.from_numpy(ws.squeeze())
        for i in range(len(graphs)):
            graph_pyg = graphs[i].to_pyg_graph()
            gp = Data(x=graph_pyg.x.cuda() if use_cuda else graph_pyg.x,
                      edge_index=graph_pyg.edge_index.cuda() if use_cuda else graph_pyg.edge_index)
            gp.y = ys[i]
            gp.w = ws[i]
            gp.smiles = smiles[i]

            new_dataset_as_list.append(gp)

    return new_dataset_as_list


def get_latest_test_metrics(test_metrics_per_epoch):
    if not test_metrics_per_epoch:
        return None, None

    test_epoch = max(test_metrics_per_epoch.keys())
    return test_epoch, test_metrics_per_epoch[test_epoch]


def infer_dataset_name(argsdict):
    if argsdict['moleculenet_dataset'] is not None:
        return argsdict['moleculenet_dataset']
    if argsdict['pyg_dataset'] is not None:
        return argsdict['pyg_dataset']
    if argsdict['custom_dataset_train'] is not None:
        return Path(argsdict['custom_dataset_train']).stem

    return 'unknown'


def sanitize_path_component(value):
    value = str(value).strip()
    value = re.sub(r'[^A-Za-z0-9._-]+', '-', value)
    return value.strip('-.') or 'unknown'


def build_run_output_dir(argsdict):
    dataset_name = sanitize_path_component(infer_dataset_name(argsdict))
    conv_type = sanitize_path_component(argsdict['conv_type'])
    readout = sanitize_path_component(argsdict['readout'])

    config_parts = [
        f'l{argsdict["num_layers"]}',
        f'h{argsdict["gnn_intermediate_dim"]}',
        f'n{argsdict["gnn_output_node_dim"]}',
        f'o{argsdict["output_nn_intermediate_dim"]}',
        f'lr{str(argsdict["learning_rate"]).replace(".", "p")}',
    ]

    if argsdict['conv_type'] == 'GATv2':
        config_parts.append(f'heads{argsdict["gat_heads"]}')
    if argsdict['readout'] == 'ours':
        config_parts.extend([
            f'wl{argsdict["walk_length"]}',
            f'wpn{argsdict["walks_per_node"]}',
        ])

    config_tag = '_'.join(sanitize_path_component(part) for part in config_parts)

    config_dir = RESULTS_ROOT / dataset_name / conv_type / readout / config_tag

    if argsdict['itr'] is not None:
        return config_dir
    if argsdict['moleculenet_random_split_seed'] is not None:
        return config_dir / f'seed{argsdict["moleculenet_random_split_seed"]}'

    return config_dir


def parse_optional_float(value):
    if value in (None, '', 'None'):
        return None

    return float(value)


def print_test_metrics(task_type, test_metrics_per_epoch):
    test_epoch, metrics = get_latest_test_metrics(test_metrics_per_epoch)
    if metrics is None:
        return

    print(f'Test paper metrics from epoch {test_epoch}:')
    if task_type == 'regression':
        mae, _, _, r2, _ = metrics
        print(f'  MAE: {mae:.6f}')
        print(f'  R2: {r2:.6f}')
    else:
        _, roc_auc, _, mcc = metrics
        if roc_auc is not None:
            print(f'  AUROC: {roc_auc:.6f}')
        print(f'  MCC: {mcc:.6f}')


def build_result_row(task_type, argsdict, test_metrics_per_epoch):
    test_epoch, metrics = get_latest_test_metrics(test_metrics_per_epoch)
    if metrics is None:
        return None

    row = {
        'dataset': infer_dataset_name(argsdict),
        'task_type': task_type,
        'row_type': 'iter' if argsdict['itr'] is not None else 'result',
        'itr': argsdict['itr'],
        'test_epoch': test_epoch,
        'mae': '',
        'r2': '',
        'auroc': '',
        'mcc': '',
    }
    row.update(filter_model_only_args(argsdict, include_in_channels=False))

    if task_type == 'regression':
        mae, _, _, r2, _ = metrics
        row.update({
            'mae': float(mae),
            'r2': float(r2),
        })
    else:
        _, roc_auc, _, mcc = metrics
        row.update({
            'auroc': '' if roc_auc is None else float(roc_auc),
            'mcc': float(mcc),
        })

    return row


def write_result_csv(out_dir, row):
    if row is None:
        return

    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)
    result_path = out_path / 'result.csv'

    existing_rows = load_result_rows(out_dir)
    normalized_row = {field: row.get(field, '') for field in RESULT_FIELDNAMES}

    if normalized_row['row_type'] == 'iter':
        iter_rows = [existing_row for existing_row in existing_rows if existing_row.get('row_type') == 'iter']
        iter_rows = [existing_row for existing_row in iter_rows if str(existing_row.get('itr', '')) != str(normalized_row['itr'])]
        iter_rows.append(normalized_row)
        iter_rows.sort(key=lambda existing_row: int(existing_row['itr']))
        rows_to_write = iter_rows + build_summary_rows(iter_rows)
    else:
        rows_to_write = [normalized_row]

    with result_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_to_write)


def load_result_rows(run_dir):
    result_path = Path(run_dir) / 'result.csv'
    if not result_path.exists():
        return []

    with result_path.open('r', newline='') as handle:
        return list(csv.DictReader(handle))


def build_summary_rows(iter_rows):
    if not iter_rows:
        return []

    template = iter_rows[0]
    summary_rows = []

    for row_type, reducer in [('mean', np.mean), ('std', np.std)]:
        summary_row = {field: template.get(field, '') for field in RESULT_FIELDNAMES}
        summary_row['row_type'] = row_type
        summary_row['itr'] = ''
        summary_row['test_epoch'] = ''

        for metric_name in ['mae', 'r2', 'auroc', 'mcc']:
            values = [parse_optional_float(iter_row.get(metric_name)) for iter_row in iter_rows]
            values = [value for value in values if value is not None]
            summary_row[metric_name] = '' if not values else float(reducer(values))

        summary_rows.append(summary_row)

    return summary_rows


def _legacy_print_iter_aggregate(task_type, out_dir):
    out_path = Path(out_dir)
    match = re.match(r'^(?P<prefix>.+)_itr(?P<itr>\d+)$', out_path.name)
    if match is None:
        return

    prefix = match.group('prefix')
    collected_metrics = []
    completed_iters = []

    for sibling in sorted(out_path.parent.glob(f'{prefix}_itr*')):
        sibling_match = re.match(rf'^{re.escape(prefix)}_itr(?P<itr>\d+)$', sibling.name)
        if sibling_match is None:
            continue

        metrics = load_latest_saved_test_metrics(sibling)
        if metrics is None:
            continue

        completed_iters.append(int(sibling_match.group('itr')))
        collected_metrics.append(metrics)

    if not collected_metrics:
        return

    print(f'Aggregate paper metrics over {len(collected_metrics)} completed iters {completed_iters}:')
    if task_type == 'regression':
        maes = [float(metrics[0]) for metrics in collected_metrics]
        r2s = [float(metrics[3]) for metrics in collected_metrics]
        print(f'  MAE: {np.mean(maes):.6f} ± {np.std(maes):.6f}')
        print(f'  R2: {np.mean(r2s):.6f} ± {np.std(r2s):.6f}')
    else:
        aurocs = [float(metrics[1]) for metrics in collected_metrics if metrics[1] is not None]
        mccs = [float(metrics[3]) for metrics in collected_metrics]
        if aurocs:
            print(f'  AUROC: {np.mean(aurocs):.6f} ± {np.std(aurocs):.6f}')
        print(f'  MCC: {np.mean(mccs):.6f} ± {np.std(mccs):.6f}')


def print_iter_aggregate(task_type, out_dir):
    iter_rows = [row for row in load_result_rows(out_dir) if row.get('row_type') == 'iter']
    if not iter_rows:
        return

    completed_iters = sorted(int(row['itr']) for row in iter_rows)

    print(f'Aggregate paper metrics over {len(iter_rows)} completed iters {completed_iters}:')
    if task_type == 'regression':
        maes = [parse_optional_float(row['mae']) for row in iter_rows]
        r2s = [parse_optional_float(row['r2']) for row in iter_rows]
        maes = [value for value in maes if value is not None]
        r2s = [value for value in r2s if value is not None]
        print(f'  MAE: {np.mean(maes):.6f} +- {np.std(maes):.6f}')
        print(f'  R2: {np.mean(r2s):.6f} +- {np.std(r2s):.6f}')
    else:
        aurocs = [parse_optional_float(row['auroc']) for row in iter_rows]
        mccs = [parse_optional_float(row['mcc']) for row in iter_rows]
        aurocs = [value for value in aurocs if value is not None]
        mccs = [value for value in mccs if value is not None]
        if aurocs:
            print(f'  AUROC: {np.mean(aurocs):.6f} +- {np.std(aurocs):.6f}')
        print(f'  MCC: {np.mean(mccs):.6f} +- {np.std(mccs):.6f}')


def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()

    # Program-level args
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--early_stopping_patience', type=int, default=30)

    # Optionally load from a saved checkpoint
    parser.add_argument('--ckpt_file', type=str)

    ### Data loading arguments ###
    # MoleculeNet dataset (required deepchem)
    parser.add_argument('--moleculenet_dataset', type=str)
    parser.add_argument('--moleculenet_random_split_seed', type=int)

    # PyTorch Geometric datasets
    parser.add_argument('--pyg_dataset', type=str)
    parser.add_argument('--pyg_dataset_splits_folder', type=str)
    parser.add_argument('--itr', type=int)

    # Custom molecular dataset
    parser.add_argument('--custom_dataset_train', type=str, required=False)
    parser.add_argument('--custom_dataset_validate', type=str, required=False)
    parser.add_argument('--custom_dataset_test', type=str, required=False)
    parser.add_argument('--custom_dataset_smiles_column', type=str, required=False)
    parser.add_argument('--custom_dataset_label_column', type=str, required=False)
    parser.add_argument('--custom_max_atomic_num', type=int, required=False)
    parser.add_argument('--custom_dataset_use_standard_scaler_on_label', dest='custom_dataset_use_standard_scaler_on_label', action='store_true',required=False)
    parser.add_argument('--custom_dataset_no_use_standard_scaler_on_label', dest='custom_dataset_use_standard_scaler_on_label', action='store_false',required=False)
    ### Data loading arguments ###
    
    # Required for both MoleculeNet and PyTorch Geometric datasets
    parser.add_argument('--dataset_download_dir', type=str)

    parser = GNN.add_model_specific_args(parser)

    # Add all the available trainer options to argparse
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()
    argsdict = vars(args)

    # Check both variables are not set at the same time
    if argsdict['moleculenet_dataset'] is not None:
        assert argsdict['pyg_dataset'] is None and argsdict['custom_dataset_train'] is None, 'Can only have a single data source active (MoleculeNet OR PyTorch Geometric OR custom molecular dataset).'
    if argsdict['pyg_dataset'] is not None:
        assert argsdict['moleculenet_dataset'] is None and argsdict['custom_dataset_train'] is None, 'Can only have a single data source active (MoleculeNet OR PyTorch Geometric OR custom molecular dataset).'
    if argsdict['custom_dataset_train'] is not None:
        assert argsdict['moleculenet_dataset'] is None and argsdict['pyg_dataset'] is None, 'Can only have a single data source active (MoleculeNet OR PyTorch Geometric OR custom molecular dataset).'
        assert argsdict['custom_dataset_smiles_column'] is not None and argsdict['custom_dataset_label_column'] is not None and argsdict['custom_max_atomic_num'] is not None \
            and argsdict['custom_dataset_use_standard_scaler_on_label'] is not None, 'Must provide all necessary custom dataset settings.'

    if argsdict['moleculenet_dataset'] is not None:
        assert argsdict['moleculenet_dataset'] in MOLNET_DS, f'MoleculeNet dataset must be within the provided list: {str(MOLNET_DS)}'

    if argsdict['pyg_dataset'] is not None:
        assert argsdict['pyg_dataset'] in PYG_DS or argsdict['pyg_dataset'] in PYG_OTHER_DS,  f'PyG dataset must be within the provided list: {str(PYG_DS + PYG_OTHER_DS)}'

    if argsdict['moleculenet_dataset']:
        assert argsdict['moleculenet_random_split_seed'] is not None, 'Must provide MoleculeNet random seed for the splits.'

    if argsdict['pyg_dataset'] and argsdict['pyg_dataset'] not in ['GNNBenchmark_MNIST', 'ZINC']:
        assert argsdict['pyg_dataset_splits_folder'] is not None and argsdict['itr'] is not None, 'Must provide PyG random splits.'

    if argsdict['custom_dataset_train'] is None:
        assert argsdict['dataset_download_dir'] is not None, 'Must provide a download directory for the datasets.'

    argsdict['out_dir'] = str(build_run_output_dir(argsdict))
    print(f'Results will be written to: {argsdict["out_dir"]}')

    assert argsdict['num_layers'] is not None and argsdict['num_layers'] > 1, 'Must provide a number of layers that is > 1.'

    if argsdict['conv_type'] == 'GATv2':
        assert argsdict['gat_heads'] is not None and argsdict['gat_dropouts'] is not None, 'Must provide the --gat_heads and --gat_dropouts arguments for GATv2.'

    # ------------
    # data
    # ------------

    ### WARINING: Some splitters (e.g. ScaffoldSplitter()) do not change according to seed

    if argsdict['moleculenet_dataset'] is not None:
        import deepchem as dc
        in_channels = 30

        if argsdict['moleculenet_dataset'] == 'QM9':
            tasks, datasets, transformers = dc.molnet.load_qm9(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                               splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.RandomSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'MAE'
            task_type = 'regression'

        elif argsdict['moleculenet_dataset'] == 'FreeSolv':
            tasks, datasets, transformers = dc.molnet.load_sampl(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                                 splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.RandomSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'MSE'
            task_type = 'regression'


        elif argsdict['moleculenet_dataset'] == 'Lipo':
            tasks, datasets, transformers = dc.molnet.load_lipo(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                                splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.RandomSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'MSE'
            task_type = 'regression'

        elif argsdict['moleculenet_dataset'] == 'HIV':
            tasks, datasets, transformers = dc.molnet.load_hiv(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                               splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.ScaffoldSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'BCEWithLogits'
            task_type = 'binary_classification'

        elif argsdict['moleculenet_dataset'] == 'BBBP':
            tasks, datasets, transformers = dc.molnet.load_bbbp(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                                splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.ScaffoldSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'BCEWithLogits'
            task_type = 'binary_classification'
        elif argsdict['moleculenet_dataset'] == 'SIDER':
            tasks, datasets, transformers = dc.molnet.load_sider(featurizer=dc.feat.MolGraphConvFeaturizer(),
                                                                 splitter=None, data_dir=argsdict["dataset_download_dir"], save_dir=argsdict["dataset_download_dir"])

            splitter = dc.splits.splitters.RandomSplitter()
            random_seed = int(argsdict['moleculenet_random_split_seed'])
            datasets = splitter.train_valid_test_split(datasets[0], frac_train=0.8, frac_valid=0.1, frac_test=0.1, seed=random_seed)

            loss_metric = 'BCEWithLogits'
            task_type = 'binary_classification'
        else:
            raise ValueError(f"Unsupported MoleculeNet dataset: {argsdict['moleculenet_dataset']}")


        train_dataset, validation_dataset, test_dataset = datasets
        train_dataset = train_dataset.make_pytorch_dataset(epochs=1, deterministic=True, batch_size=len(train_dataset))
        validation_dataset = validation_dataset.make_pytorch_dataset(epochs=1, deterministic=True, batch_size=len(validation_dataset))
        test_dataset = test_dataset.make_pytorch_dataset(epochs=1, deterministic=True, batch_size=len(test_dataset))

        train_dataset = deepchem_iterable_dataset_to_tensors(train_dataset, use_cuda=argsdict['gpus'] == 1)
        validation_dataset = deepchem_iterable_dataset_to_tensors(validation_dataset, use_cuda=argsdict['gpus'] == 1)
        test_dataset = deepchem_iterable_dataset_to_tensors(test_dataset, use_cuda=argsdict['gpus'] == 1)

        num_train_workers = (0, 0, 0)
        train_loader = DataLoader(train_dataset, batch_size=argsdict['batch_size'], num_workers=num_train_workers[0])
        validation_loader = DataLoader(validation_dataset, batch_size=argsdict['batch_size'], num_workers=num_train_workers[1])
        test_loader = DataLoader(test_dataset, batch_size=argsdict['batch_size'], num_workers=num_train_workers[2])

        num_tasks = len(tasks)

    elif argsdict['pyg_dataset'] == 'GNNBenchmark_MNIST':
        import torch_geometric.datasets as ds
        root = argsdict['dataset_download_dir']

        train_dataset = ds.GNNBenchmarkDataset(root=root, name='MNIST', split='train')
        validation_dataset = ds.GNNBenchmarkDataset(root=root, name='MNIST', split='val')
        test_dataset = ds.GNNBenchmarkDataset(root=root, name='MNIST', split='test')

        loss_metric = 'CrossEntropyLoss'
        task_type = 'multi_classification'

        in_channels = train_dataset.num_node_features
        num_tasks = train_dataset.num_classes

        train_loader = DataLoader(train_dataset, batch_size=argsdict['batch_size'], shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=argsdict['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=argsdict['batch_size'])

        print(f'Using {argsdict["pyg_dataset"]} dataset from PyTorch Geometric.')

    elif argsdict['pyg_dataset'] is not None and argsdict['pyg_dataset'] == 'ZINC':
        import torch_geometric.datasets as ds
        root = argsdict['dataset_download_dir']

        train_dataset = ds.ZINC(root=root, split='train', subset=False)
        validation_dataset = ds.ZINC(root=root, split='val', subset=False)
        test_dataset = ds.ZINC(root=root, split='test', subset=False)

        loss_metric = 'MSE'
        task_type = 'regression'

        in_channels = train_dataset.num_node_features
        num_tasks = 1

    elif argsdict['pyg_dataset'] is not None and argsdict['pyg_dataset'] in ['ENZYMES', 'github_stargazers', 'reddit_threads',
                                                                              'SYNTHETIC', 'SYNTHETICnew', 'Synthie',
                                                                              'Cuneiform', 'IMDB-BINARY', 'MUTAG',
                                                                              'Mutagenicity']:
        import torch_geometric.datasets as ds
        root = argsdict['dataset_download_dir']

        dir_path = os.path.join(argsdict['pyg_dataset_splits_folder'], argsdict['pyg_dataset'])
        perm = np.load(os.path.join(dir_path, f'random_permutation_{argsdict["itr"]}.npy'))

        dataset = ds.TUDataset(root=root, name=argsdict['pyg_dataset'], use_node_attr=True)

        if argsdict['pyg_dataset'] in ['github_stargazers', 'IMDB-BINARY', 'reddit_threads']:
            degree_0 = np.max([np.max(degree(d.edge_index[0]).detach().cpu().numpy()) for d in dataset])
            degree_1 = np.max([np.max(degree(d.edge_index[1]).detach().cpu().numpy()) for d in dataset])
            dataset = ds.TUDataset(root=root, name=argsdict['pyg_dataset'],
                                   transform=T.OneHotDegree(max_degree=int(max(degree_0, degree_1))))
            in_channels = int(max(degree_0, degree_1)) + 1
        else:
            in_channels = dataset.num_node_features

        dataset = dataset.index_select(perm)

        dataset_as_numpy = np.asarray(dataset, dtype=object)
        train, validate, test = np.split(dataset_as_numpy, [int(0.8 * len(dataset_as_numpy)), int(0.9 * len(dataset_as_numpy))])

        train_dataset = [Data(**{item[0]: item[1] for item in data}) for data in train]
        validation_dataset = [Data(**{item[0]: item[1] for item in data}) for data in validate]
        test_dataset = [Data(**{item[0]: item[1] for item in data}) for data in test]

        if argsdict['pyg_dataset'] in ['github_stargazers', 'IMDB-BINARY', 'reddit_threads', 'SYNTHETIC',
                                       'SYNTHETICnew', 'Mutagenicity', 'MUTAG']:
            task_type = 'binary_classification'
            loss_metric = 'BCEWithLogits'

        else:
            task_type = 'multi_classification'
            loss_metric = 'CrossEntropyLoss'

        num_tasks = dataset.num_classes
        if num_tasks == 2:
            num_tasks = 1

    elif argsdict['custom_dataset_train'] is not None:
        from utils.data_loading import GeometricDataModule
        custom_dataset = GeometricDataModule(batch_size=argsdict['batch_size'], seed=0,
                                            train_path=argsdict['custom_dataset_train'],
                                            separate_valid_path=argsdict['custom_dataset_validate'],
                                            separate_test_path=argsdict['custom_dataset_test'],
                                            split_train=False, num_cores=(0, 0, 0),
                                            smiles_column_name=argsdict['custom_dataset_smiles_column'],
                                            label_column_name=argsdict['custom_dataset_label_column'],
                                            use_standard_scaler=argsdict['custom_dataset_use_standard_scaler_on_label'],
                                            max_atomic_num=argsdict['custom_max_atomic_num'])

        custom_dataset.prepare_data()
        custom_dataset.setup()
        train_loader = custom_dataset.train_dataloader()
        validation_loader = custom_dataset.val_dataloader()
        test_loader = custom_dataset.test_dataloader()
        num_tasks = custom_dataset.label_dims
        in_channels = argsdict['custom_max_atomic_num'] + 27

        loss_metric = 'MSE'
        task_type = 'regression'
        num_tasks = 1
        train_dataset = custom_dataset.dataset


    if argsdict['pyg_dataset']:
        train_loader = DataLoader(train_dataset, batch_size=argsdict['batch_size'], shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=argsdict['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=argsdict['batch_size'])

    print(f'Size of training dataset = {len(train_loader)}.')
    if validation_loader is not None:
        print(f'Size of validation dataset = {len(validation_loader)}.')
    if test_loader is not None:
        print(f'Size of test dataset = {len(test_loader)}.')


    # ------------
    # model
    # ------------
    print('Creating model...')
    model = GNN(in_channels=in_channels,
        **filter_model_only_args(argsdict, include_in_channels=False), output_nn_out_dim=num_tasks, loss_metric=loss_metric,
        task_type=task_type)

    print('Model summary: ')
    print(model)

    # ------------
    # training
    # ------------

    monitor = 'validation_total_loss' if argsdict['custom_dataset_train'] is None else 'train_total_loss'

    if argsdict['custom_dataset_train'] is None:
        early_stopping = EarlyStopping(
                monitor=monitor,
                min_delta=0.00,
                patience=argsdict['early_stopping_patience'],
                verbose=False,
                mode='min'
            )

    callbacks = [early_stopping] if argsdict['custom_dataset_train'] is None else []

    print('Creating Trainer...')
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=False,
        enable_checkpointing=False,
    )

    print('Starting training...')

    loaders = (train_loader, validation_loader) if argsdict['custom_dataset_train'] is None else (train_loader,)

    if not argsdict['ckpt_file']:
        trainer.fit(model, *loaders)
    else:
        trainer.fit(model, *loaders, ckpt_path=argsdict['ckpt_file'])


    # ------------
    # testing
    # ------------
    should_test_flag = (argsdict['moleculenet_dataset'] is not None) or (argsdict['pyg_dataset'] is not None) or (argsdict['custom_dataset_train'] is not None and argsdict['custom_dataset_test'] is not None)
    if should_test_flag:
        trainer.test(model, dataloaders=test_loader)
        print_test_metrics(task_type=task_type, test_metrics_per_epoch=model.test_metrics_per_epoch)

    # ------------
    # saving
    # ------------
    if should_test_flag:
        write_result_csv(
            out_dir=argsdict['out_dir'],
            row=build_result_row(
                task_type=task_type,
                argsdict=argsdict,
                test_metrics_per_epoch=model.test_metrics_per_epoch,
            ),
        )
        print_iter_aggregate(task_type=task_type, out_dir=argsdict['out_dir'])


if __name__ == '__main__':
    main()
