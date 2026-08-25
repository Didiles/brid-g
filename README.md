# BRID-G (plug-and-play)

Build an undirected **BRID-G** graph from a feature matrix \(X \in \mathbb{R}^{N \times D}\).

No dataset loaders, no GNN training — only graph construction. Suitable as a preprocessing step before any graph neural network.

## Install

```bash
git clone https://github.com/Didiles/brid-g.git
cd brid-g
pip install -e .
# or
pip install -r requirements.txt
```

Dependencies: `numpy`, `scikit-learn`, `networkx`.

## Quick start

```python
import numpy as np
from bridg import build_brid_g, BridGConfig

# Your embeddings / feature matrix (N, D)
features = np.random.randn(500, 128)

# Paper defaults: k=40, mandatory=5, RBO θ=0.1, triangulation=5,
# Euclidean distance, L2 normalization for ranking only
edge_index, G = build_brid_g(features)

# edge_index: np.ndarray with shape (2, E), dtype int64
# G: networkx.Graph (undirected)
print(G.number_of_nodes(), G.number_of_edges())
```

Custom hyperparameters:

```python
cfg = BridGConfig(
    k=40,
    mandatory=5,
    rbo_threshold=0.1,
    triangulation=5,
    distance_metric="euclidean",  # or "cosine"
    normalize_l2=True,
    reciprocal=True,
    similarity="rbo",             # or "jaccard"
)
edge_index, G = build_brid_g(features, config=cfg)

# NumPy only (no NetworkX object)
edge_index = build_brid_g(features, config=cfg, return_networkx=False)
```

See [`examples/basic_usage.py`](examples/basic_usage.py).

## Pipeline

Canonical executed order (aligned with the reference implementation and thesis Cap. 4):

1. **Optional L2** — normalize features **only for ranking / influence** (keep the original `X` for your classifier).
2. **Rankings** — top-`k` nearest neighbours per node; the query itself is excluded.
3. **Influence selection** — for every neighbour used as an anchor, keep diverse candidates via influence dominance; materialize an edge only if the pair passes **reciprocity** and **RBO ≥ θ** (when enabled).
4. **Triangulation** — connect pairs among the first `t` accepted neighbours (insertion order), still subject to the same filter.
5. **Mandatory edges** — force the top-`m` ranked neighbours **without** reciprocity / RBO.
6. **Assemble** — undirected graph (`networkx.Graph` union of directed stubs).

```
X  →  L2 (opt.)  →  top-k rankings
   →  influence + filter  →  triangulation  →  mandatory  →  G
```

## Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `k` | 40 | Neighbourhood size (self excluded) |
| `mandatory` | 5 | Top-`m` forced edges after triangulation (`None` to disable) |
| `rbo_threshold` | 0.1 | Min neighbourhood similarity (`None` disables similarity gate) |
| `triangulation` | 5 | Elite size for triangulation (`None` / `≤1` disables) |
| `distance_metric` | `"euclidean"` | `"euclidean"` or `"cosine"` |
| `normalize_l2` | `True` | L2 for ranking only |
| `reciprocal` | `True` | Require mutual top-`k` membership |
| `similarity` | `"rbo"` | `"rbo"` or `"jaccard"` |
| `rbo_p` | 0.9 | RBO persistence (truncated residual sum) |

Influence:

- Euclidean: \(I(u,v) = 1 / d_2(u,v)\), diagonal 0
- Cosine: \(I(u,v) = 1 - d_{\cos}(u,v)\), diagonal 0

RBO is the truncated residual form to depth `k` with persistence `p` (default 0.9), matching the reference code used in the experiments.

## API

| Symbol | Role |
|--------|------|
| `build_brid_g(features, config=None, return_networkx=True)` | Main entry point |
| `BridGConfig` | Frozen dataclass of hyperparameters |
| `compute_rbo` / `compute_jaccard` | Neighbourhood similarity helpers |

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Parity tests compare undirected edge sets against `brid_g.graphs.create_graph_brid_n_rnn` when that package is importable. They are skipped otherwise.

## Citation

If you use BRID-G, please cite the related publications (SIBGRAPI / WTDBD) from the BRID-G project.

## License

MIT — see [LICENSE](LICENSE).
