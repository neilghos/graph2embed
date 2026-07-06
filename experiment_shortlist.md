# Experiment Shortlist

Source:
- `Appendix (1).pdf`
- upstream supplementary CSVs / split files in `gnn-neural-readouts-main`

## T.11 Small Molecules

Keep:
- `MUTAG`
- `Mutagenicity`

Drop:
- `AIDS`
- `FRANKENSTEIN`

Rationale:
- `MUTAG` is tiny and fast.
- `Mutagenicity` is larger but still manageable and less saturated than `AIDS`.
- `AIDS` is too saturated for a useful paper comparison.
- `FRANKENSTEIN` is dropped to keep T.11 chemistry-focused.

## T.10 Computer Vision

Keep:
- `MNIST`

Notes:
- current computer-vision pick is `MNIST`
- `CIFAR10` not selected yet

## T.7 Regression

Keep:
- `ZINC`

Notes:
- `T.7` is only `ZINC`
- use it as the compact graph-regression benchmark from this appendix block
- unlike the `itr0..4` PyG shuffled datasets, `ZINC` uses the provided train/validation/test split

## T.6 TUDataset Computer Vision

Keep:
- `Cuneiform`

Notes:
- `Cuneiform` is the smallest T.6 dataset
- size: `267` graphs
- larger alternatives in the same block:
  - `COIL-RAG`: `3900`
  - `COIL-DEL`: `3900`

## T.5 Bioinformatics Classification

Keep:
- `ENZYMES`

Notes:
- `ENZYMES` is the smaller T.5 dataset
- size: `600` graphs
- other dataset in the block:
  - `PROTEINS_full`: `1113`

## T.4 Synthetic Classification

Keep:
- `SYNTHETIC`
- `SYNTHETICnew`
- `Synthie`

Notes:
- smallest T.4 datasets:
  - `SYNTHETIC`: `300`
  - `SYNTHETICnew`: `300`
  - `Synthie`: `400`
- larger T.4 datasets not selected:
  - `COLORS-3`: `10500`
  - `TRIANGLES`: `45000`

## T.3 TUDataset Social Networks

Keep:
- `reddit_threads`
- `github_stargazers`
- `IMDB-BINARY`

Notes:
- use `reddit_threads` as the large-scale pick
  - size: `203088` graphs
  - avg nodes: `23.93`
  - avg edges: `24.99`
- use `github_stargazers` as the small/medium pick
  - size: `12725` graphs
  - avg nodes: `113.79`
  - avg edges: `234.64`
- use `IMDB-BINARY` as the development-small pick
  - size: `1000` graphs
  - avg nodes: `19.77`
  - avg edges: `96.53`
- larger or alternative T.3 options not selected:
  - `REDDIT-BINARY`: `2000` graphs, but much larger per-graph size (`429.63` nodes avg)
  - `REDDIT-MULTI-12K`: `11929` graphs, also very large per-graph size (`391.41` nodes avg)
  - `twitch_egos`: `127094`
  - `TWITTER-Real-Graph-Partial`: `144033`

## T.2 MoleculeNet Classification

Keep:
- `HIV`
- `BBBP`
- `SIDER`

Notes:
- selected T.2 trio:
  - `HIV` as the practical large-scale pick
    - size: `41127`
    - binary classification, `1` task
  - `BBBP` as the small/medium pick
    - size: `2039`
    - binary classification, `1` task
  - `SIDER` as the extra multitask classification pick
    - size: `1396`
    - classification, `27` tasks
- datasets not selected by default:
  - `PCBA`: `437918` graphs, `128` tasks
    - useful only if you explicitly want a very large multitask stress test
  - `BACE_CLS`: `1513`
    - binary classification, `1` task

## T.1 MoleculeNet Regression

Keep:
- `QM9`
- `Lipophilicity`
- `FreeSolv`

Notes:
- selected T.1 trio:
  - `QM9` as the core large-scale regression benchmark
    - size: `132480`
    - regression, `12` tasks
    - also the main paper table dataset, so it has to stay
  - `Lipophilicity` as the small/medium pick
    - size: `4200`
    - regression, `1` task
  - `FreeSolv` as the development-small pick
    - size: `639`
    - regression, `1` task
- datasets not selected by default:
  - `QM8`: `21747`, regression, `12` tasks
    - useful if you want a second larger quantum benchmark, but mostly redundant once `QM9` is kept
  - `QM7`: `6834`, regression, `1` task
    - possible backup if you want another quantum-mechanics dataset
  - `ESOL`: `1127`, regression, `1` task
    - similar role to `FreeSolv`, but slightly larger
  - `BACE_REGR`: `1513`, regression, `1` task
    - optional if you specifically want the paired regression/classification BACE setting

## Skipped Groups

- `T.8` skipped
- `T.9` skipped

## Notes

- Current goal: simplify the experiment setup and keep only a compact, paper-friendly benchmark list.
- Next step: pick 2 datasets each for the remaining appendix groups (`T.3 ... T.N`) with preference for small, non-saturated, quick-to-run datasets.
