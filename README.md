BRID-G — **Better Results with Influence Diversification for Graphs**
for Graph Neural Networks:

GCNs need two inputs: features and a graph. In citation networks the
graph already exists. In images, it has to be built — usually from
similarity (kNN / reciprocal kNN), which ignores diversity and can
leave neighborhoods full of redundant neighbors.

This repo is the plug-and-play construction of that graph. You pass a
feature matrix $X \in \mathbb{R}^{N \times D}$ and get an undirected
BRID-G graph. No dataset loaders, no GNN training.

## 📝 Project Summary

The usual recipe for image graphs is: extract embeddings, connect each
sample to its $k$ nearest neighbors (sometimes only if the pair is
reciprocal). That is similarity-only. Homogeneous neighborhoods limit
what the GCN can propagate and tend to make oversmoothing worse.

BRID-G is an adaptation of BRID [Santos et al. 2013] — originally
from similarity search with diversity — to GCN graph construction.
The idea is to prune mutually redundant edges inside a ranked kNN
neighborhood, instead of treating every neighbor the same.

In short, $(X, k, \text{hyperparameters}) \mapsto E$ in three stages:

1. **Influence filtering** — prune redundancies inside the top-$k$
   (influence dominance).
2. **Triangulation** — put some local cohesion back, among the first
   $t$ accepted neighbors, still under the same filter.
3. **Structural filtering** — keep an edge only if the pair is
   reciprocal and the ranked lists agree (RBO or Jaccard, $\geq \theta$).
   A small set of mandatory top-$m$ edges is added without that filter.

Canonical order in this implementation (thesis Cap. 4):

```
X  →  L2 (opt.)  →  top-k rankings
   →  influence + filter  →  triangulation  →  mandatory  →  G
```

L2, when enabled, is only for ranking / influence. Keep the original
`X` for the classifier.

Distance / influence (`distance_metric`). Diagonal of $I$ is always 0.

**Euclidean**

$$
d_2 : \mathbb{R}^D \times \mathbb{R}^D \to [0, +\infty),
\quad
d_2(u,v) = \lVert u - v \rVert_2
$$

$$
I : \mathbb{R}^D \times \mathbb{R}^D \to [0, +\infty),
\quad
I(u,v) = 1 / d_2(u,v)
$$

**Cosine** (scikit-learn: $d_{\cos} = 1 - \cos\theta$)

$$
d_{\cos} : \mathbb{R}^D \times \mathbb{R}^D \to [0, 2],
\quad
d_{\cos}(u,v) = 1 - \frac{u \cdot v}{\lVert u \rVert_2 \,\lVert v \rVert_2}
$$

$$
I : \mathbb{R}^D \times \mathbb{R}^D \to [-1, 1],
\quad
I(u,v) = 1 - d_{\cos}(u,v)
$$

RBO is the truncated residual form to depth `k` with persistence `p`
(default 0.9), same as the code used in the experiments.

## 📂 Project Structure

```
/
├── bridg/                 # Package
│   ├── builder.py         # build_brid_g
│   ├── config.py          # BridGConfig
│   ├── ranking.py         # Distances, influence, kNN
│   ├── similarity.py      # RBO / Jaccard
│   ├── stages.py          # Influence, triangulation, mandatory
│   └── io.py              # Undirected graph
├── examples/              # Minimal usage
├── tests/                 # pytest (+ parity vs. reference, if available)
├── pyproject.toml
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

**1. Clone:**
```bash
git clone https://github.com/Didiles/brid-g.git
cd brid-g
```

**2. Virtual environment (recommended):**
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install:**
```bash
pip install -e .
# or
pip install -r requirements.txt
```

Depends on `numpy`, `scikit-learn`, `networkx`.

## How to Run

Everything goes through `BridGConfig` and `build_brid_g`. Paper
defaults: `k=40`, `mandatory=5`, RBO $\theta=0.1$, `triangulation=5`,
Euclidean, L2 only for ranking.

```python
import numpy as np
from bridg import build_brid_g, BridGConfig

# Embeddings / feature matrix (N, D)
features = np.random.randn(500, 128)

edge_index, G = build_brid_g(features)
# edge_index: (2, E), int64
# G: networkx.Graph (undirected)
print(G.number_of_nodes(), G.number_of_edges())
```

If you need to change the knobs:

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

# NumPy only
edge_index = build_brid_g(features, config=cfg, return_networkx=False)
```

Or just:

```bash
python examples/basic_usage.py
```

### Parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `k` | 40 | Neighborhood size (self excluded) |
| `mandatory` | 5 | Top-`m` forced edges after triangulation (`None` to disable) |
| `rbo_threshold` | 0.1 | Min neighborhood similarity (`None` disables the gate) |
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

Parity tests compare the undirected edge set against
`brid_g.graphs.create_graph_brid_n_rnn` when that package is
importable. Otherwise they are skipped.

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
