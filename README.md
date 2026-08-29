BRID-G — **Better Results with Influence Diversification for Graphs**
for Graph Neural Networks:

Repository for the plug-and-play implementation of BRID-G, a graph
construction method that combines neighborhood ranking, influence
dominance, reciprocity, and rank-biased overlap to better capture
local structure before any Graph Neural Network.

Build an undirected BRID-G graph from a feature matrix $X \in \mathbb{R}^{N \times D}$.
No dataset loaders, no GNN training — only graph construction. Suitable
as a preprocessing step before any graph neural network.

## 📝 Project Summary

In image classification and related tasks, graphs are usually
constructed from feature similarity (typically *k*-NN), where treating
all neighbors equally may overlook important variations in relevance
and produce redundant edges. Motivated by this research gap, we
propose BRID-G (Better Results with Influence Diversification for Graphs).

To achieve this, the method ranks the top-*k* nearest neighbors of each
sample, then selects diverse candidates via influence dominance.
An edge is materialized only if the pair passes reciprocity and a
neighborhood-similarity gate (RBO or Jaccard). Triangulation connects
pairs among the first accepted neighbors (still subject to the same
filter), and a small set of mandatory top-ranked edges is added
without filtering. The result is an undirected graph.

Canonical executed order (aligned with the reference implementation
and thesis Cap. 4):

1. **Optional L2** — normalize features **only for ranking / influence**
   (keep the original `X` for your classifier).
2. **Rankings** — top-`k` nearest neighbours per node; the query itself
   is excluded.
3. **Influence selection** — for every neighbour used as an anchor,
   keep diverse candidates via influence dominance; materialize an
   edge only if the pair passes **reciprocity** and **RBO ≥ θ**
   (when enabled).
4. **Triangulation** — connect pairs among the first `t` accepted
   neighbours (insertion order), still subject to the same filter.
5. **Mandatory edges** — force the top-`m` ranked neighbours **without**
   reciprocity / RBO.
6. **Assemble** — undirected graph (`networkx.Graph` union of directed
   stubs).

```
X  →  L2 (opt.)  →  top-k rankings
   →  influence + filter  →  triangulation  →  mandatory  →  G
```

Influence:

- Euclidean: $I(u,v) = 1 / d_2(u,v)$, diagonal 0
- Cosine: $I(u,v) = 1 - d_{\cos}(u,v)$, diagonal 0

RBO is the truncated residual form to depth `k` with persistence `p`
(default 0.9), matching the reference code used in the experiments.

## 📂 Project Structure

The code has been modularized to ensure clarity and maintainability,
following the structure below:

```
/
├── bridg/                 # Package: pipeline and public API
│   ├── builder.py         # Main entry point (build_brid_g)
│   ├── config.py          # BridGConfig hyperparameters
│   ├── ranking.py         # Distances, influence, kNN rankings
│   ├── similarity.py      # RBO / Jaccard neighborhood similarity
│   ├── stages.py          # Influence, triangulation, mandatory
│   └── io.py              # Undirected graph assembly
├── examples/              # Minimal usage example
├── tests/                 # pytest (+ parity vs. reference package)
├── pyproject.toml
├── requirements.txt
└── README.md              # This file
```

## ⚙️ Installation and Setup

Follow the steps below to set up the environment and run the project.

**1. Clone the repository:**
```bash
git clone https://github.com/Didiles/brid-g.git
cd brid-g
```

**2. Create and activate a virtual environment (recommended):**
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install the dependencies:**
```bash
pip install -e .
# or
pip install -r requirements.txt
```

Dependencies: `numpy`, `scikit-learn`, `networkx`.

## How to Run

Graph construction is controlled through `BridGConfig` and executed
by `build_brid_g`. Paper defaults are `k=40`, `mandatory=5`,
RBO θ=`0.1`, `triangulation=5`, Euclidean distance, and L2
normalization for ranking only.

**Example:**
```python
import numpy as np
from bridg import build_brid_g, BridGConfig

# Your embeddings / feature matrix (N, D)
features = np.random.randn(500, 128)

# Paper defaults
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

After setting the desired parameters, you can also run the example
from the project's root folder:

```bash
python examples/basic_usage.py
```

### Parameters

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

### Tests

```bash
pip install -e ".[dev]"
pytest
```

Parity tests compare undirected edge sets against
`brid_g.graphs.create_graph_brid_n_rnn` when that package is
importable. They are skipped otherwise.

## 🎓 Author and Acknowledgements

* **Authors:** D. G. de Paulo, L. P. Valem

This work was supported by the São Paulo Research Foundation
(FAPESP, grant #2025/10602-5), the University of São Paulo
(PRPI Ordinance No. 1032, “Apoio aos Novos Docentes”), and
the Coordination for the Improvement of Higher Education
Personnel – Brazil (CAPES) – Finance Code 001.

## Citation

If you use BRID-G, please cite the related publications (SIBGRAPI / WTDBD)
from the BRID-G project.

## License

MIT — see [LICENSE](LICENSE).
