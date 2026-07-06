# Local Validation Against Upstream

This repo copy is validated as a trimmed, experiment-focused subset of the upstream
`davidbuterez/gnn-neural-readouts` NeurIPS 2022 codebase.

## Scope Preserved From Upstream

- The active GNN backbones are the retained upstream subset:
  - `GCN`
  - `GATv2`
  - `GIN`
- The standard baseline readouts still behave as upstream for the retained subset:
  - `sum`
  - `mean`
  - `max`
- The selected benchmark datasets still use upstream loading and split logic:
  - MoleculeNet:
    - `QM9`: `RandomSplitter`, `MAE`
    - `FreeSolv`: `RandomSplitter`, `MSE`
    - `Lipo`: `RandomSplitter`, `MSE`
    - `HIV`: `ScaffoldSplitter`, `BCEWithLogits`
    - `BBBP`: `ScaffoldSplitter`, `BCEWithLogits`
    - `SIDER`: `RandomSplitter`, `BCEWithLogits`
  - PyG provided-split datasets:
    - `ZINC`
    - `GNNBenchmark_MNIST`
  - PyG shuffled datasets:
    - `ENZYMES`
    - `github_stargazers`
    - `reddit_threads`
    - `SYNTHETIC`
    - `SYNTHETICnew`
    - `Synthie`
    - `Cuneiform`
    - `IMDB-BINARY`
    - `MUTAG`
    - `Mutagenicity`
- The PyG shuffled datasets still rely on upstream supplementary split artifacts in:
  - `Supplementary materials/Supplementary File 1/pyg_shuffled_datasets`
- The upstream README explicitly states that `Supplementary materials` contains:
  - random seeds and splits for datasets
  - detailed metrics tables

## Local Changes Relative to Upstream

- The repo is intentionally reduced to the selected shortlist datasets only.
- The non-selected upstream dataset branches were removed from `code/run.py`.
- The only non-upstream readout path is `readout == "ours"`.
  - Standard `sum/mean/max` still use the normal upstream pooling path.
- Extra compatibility/runtime shims were added:
  - `code/wandb.py`
  - `code/tensorflow.py`
  - `code/jax.py`
  - SSL certificate setup in `code/run.py`
  - NumPy runtime guard in `code/run.py`
- Logging/output changes are wrapper-level only:
  - print/log only paper metrics
  - configurable early stopping patience
  - sibling-iteration aggregate printing

## Artifacts Removed On Purpose

- `UMAP_graph_embeddings`
- `Supplementary materials/Supplementary File 2`
- `code/pubchem_datasets`
- `code/__pycache__`

These are not required for the retained benchmark suite.

## Important Note

- `QM9_permutations` is *not* required for normal QM9 benchmark training/evaluation.
- It is only needed for the separate QM9 node-permutation robustness analysis from the paper appendix.
- Therefore the retained QM9 benchmark pipeline remains upstream-equivalent without it.

## Conclusion

For the retained experiment suite, the local runner/model stack is upstream-equivalent
in data loading, split behavior, backbone selection, and baseline pooling behavior,
with only:

- dataset-scope reduction
- environment compatibility shims
- the added `ours` readout branch
- logging/reporting convenience changes

This local suite can be treated as a validated NeurIPS-derived experiment subset.
