# LoNALP

**LoNALP** (Local Optimization-based Network Alignment and Link Prediction) is a
unified iterative framework that jointly performs network alignment and link
prediction on protein–protein interaction (PPI) networks.

Network alignment and link prediction are mutually reinforcing: aligning
cross-species networks transfer functional-ortholog information that helps
cross-species link prediction, while newly predicted interactions in turn
refine the alignment. LoNALP harnesses this bidirectional synergy through a
closed loop:

1. **Weighted union network** — the two PPI networks are merged with the
   cross-network alignment edges, using sequence/structural similarity as the
   edge weight.
2. **Weighted node2vec** — topology-aware node embeddings are learned on the
   union network with a weight-modulated transition scheme (return / in-out
   parameters `p`, `q`).
3. **Local optimization (KM)** — the global alignment is refined via the
   Kuhn–Munkres (Hungarian) max-weight assignment, which yields a globally
   consistent one-to-one mapping that is robust to structural heterogeneity.
4. **Link prediction** — a logistic-regression classifier predicts missing
   links from the node embeddings, and the predicted links are fed back to
   update both the alignment and the embeddings in the next iteration.

## Repository layout

```
.
├── README.md
├── src/
│   ├── LoNALP.py          # main iterative algorithm (alignment + link prediction)
│   ├── node2vec_embed.py  # union-network construction + weighted node2vec
│   ├── model.py           # structural (PP-distance) similarity
│   ├── utils.py           # data loading utilities
│   ├── demo.py            # command-line entry point (runs on bigtoy)
│   └── ablation.py        # component ablation study
├── graph/                 # edge lists of G1 and G2 (bigtoy)
├── attribute/             # node attribute matrices (bigtoy)
└── alignment/             # ground-truth alignment (bigtoy)
```

## Requirements

- Python 3.7+
- `numpy`, `pandas`, `networkx`, `scipy`
- `scikit-learn` (logistic regression, cosine similarity)
- `gensim` (Word2Vec / node2vec skip-gram training)

```bash
pip install numpy pandas networkx scipy scikit-learn gensim
```

## Quick start

Run the reproduction on the toy dataset `bigtoy` from the `src/` directory:

```bash
cd src
python demo.py
```

This loads the two toy networks, runs the LoNALP iteration, and prints the
per-iteration alignment accuracy. The toy dataset is a small (150-node)
synthetic case used only to verify the pipeline end-to-end.

### Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_folder` | `../graph/` | directory holding `<name>1.edges`, `<name>2.edges` |
| `--attribute_folder` | `../attribute/` | directory holding `<name>attr1.csv`, `<name>attr2.csv` |
| `--alignment_folder` | `../alignment/` | directory holding `<name>.csv` (ground truth) |
| `--filename` | `bigtoy` | dataset name prefix |
| `--alpha` | `5` | structural-similarity sharpness (`exp(-alpha * dist)`) |
| `--layer` | `3` | neighborhood depth for structural similarity |
| `--q` | `0.5` | (legacy) cross-graph walk probability |
| `--c` | `0.5` | attribute / structure fusion weight |

## Core API

```python
from LoNALP import LoNALP

S, precision, seed1, seed2, best_acc = LoNALP(
    G1, G2, q, attr1, attr2, attribute,
    alignment_dict, alignment_dict_reversed,
    layer=3, align_train_prop=0.0, alpha=5, c=0.5, multi_walk=False,
    use_km=True,          # Kuhn-Munkres alignment (False -> greedy)
    use_feedback=True,    # feed predicted links back (False -> single round)
    p=10, q2=2,           # weighted node2vec return / in-out params
    max_iters=6,
    n_walks=20,           # walks per node
    walk_len=80,          # walk length
)
```

`use_km`, `use_feedback`, and `p`/`q2` expose the three main components for
ablation (see `ablation.py`).

## Ablation study

```bash
cd src
python ablation.py
```

Isolates the contribution of (a) the Kuhn–Munkres local optimization,
(b) the weighted node2vec embedding, and (c) the iterative feedback loop.

## Data format

- **Edges**: `<name>1.edges` / `<name>2.edges` — two-column edge lists.
- **Attributes**: `<name>attr1.csv` / `<name>attr2.csv` — per-node attribute
  matrices (row index = node id).
- **Alignment**: `<name>.csv` — two-column ground-truth node correspondence.

Node ids of the two networks must be disjoint; when constructing the union
network, G2 nodes are addressed with an offset of `max(G1 node id) + 1`.

## Reference

The method is described in the manuscript **"Local optimization-based joint
protein-protein interaction prediction and network alignment"**.
