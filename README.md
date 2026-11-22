BRID-G — **Better Results with Influence Diversification for Graphs**
for Graph Neural Networks:

GCNs need two inputs: features and a graph. In citation networks the
graph already exists. In images, it has to be built — usually from
similarity (kNN / reciprocal kNN), which ignores diversity and can
leave neighborhoods full of redundant neighbors.

This repo is the plug-and-play construction of that graph. You pass a
feature matrix $X \in \mathbb{R}^{N \times D}$ and get a BRID-G graph.
No dataset loaders, no GNN training.

## 📝 Project Summary

The usual recipe for image graphs is: extract embeddings, connect each
sample to its $k$ nearest neighbors (sometimes only if the pair is
reciprocal). That is similarity-only. Homogeneous neighborhoods limit
what the GCN can propagate and tend to make oversmoothing worse.

BRID-G is an adaptation of BRID [Santos et al. 2013] — originally
from similarity search with diversity — to GCN graph construction.
The idea is to prune mutually redundant edges inside a ranked kNN
neighborhood, instead of treating every neighbor the same.

In short, $(X, k, \text{hyperparameters}) \mapsto E$ in the order
`build_brid_g` runs:

1. **Ranking** — optional L2, then pairwise distance, influence, and
   the top-$k$ lists (self excluded).
2. **Anchors** — which of those neighbors are dominance references.
3. **Influence** — for each anchor, every other candidate is tested
   for bilateral dominance, including the nearest. A candidate is kept
   in that pass only when neither the anchor nor an already retained
   neighbor dominates it. The edge is written only if the pair is in
   each other's top-$k$ (mutual reciprocity) and the ranked lists
   agree (RBO or Jaccard, $\geq \theta$).
4. **Triangulation** — pairs among the first $t$ accepted neighbors,
   under the same filter.
5. **Mandatory** — the top-$m$ neighbors, with no reciprocity or
   similarity filter.
6. **Assembly** — `edge_index` is the directed list: both directions
   may appear, and the same ordered pair may repeat. The NetworkX
   graph is the undirected simple graph on those edges.

```
X  →  L2 (opt.)  →  top-k rankings  →  anchors
   →  influence + filter  →  triangulation + filter
   →  mandatory  →  edge_index, G
```

L2, when enabled, is only for ranking / influence. Keep the original
`X` for the classifier.

Distance / influence (`distance_metric`). Diagonal of $I$ is always 0.
Selection (dominance, triangulation, reciprocity / RBO) is the same for
every metric: it compares influence values. Rankings still sort distance
ascending.

**Lp family** — `euclidean` ($p=2$), `manhattan` ($p=1$), `minkowski`
(order `minkowski_p`, default 3), and `chebyshev` ($p=\infty$).

$$
d_p(u,v) = \lVert u - v \rVert_p,
\quad
I(u,v) = 1 / d_p(u,v)
$$

$$
d_\infty(u,v) = \max_i |u_i - v_i|,
\quad
I(u,v) = 1 / d_\infty(u,v)
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

**Correlation** (scikit-learn: $d_{\mathrm{corr}} = 1 - \mathrm{corr}$).
Rankings use that distance. Influence is the Pearson correlation itself.

$$
I(u,v) = \mathrm{corr}(u,v)
$$

A constant feature row makes correlation undefined and raises `ValueError`.

For every dissimilarity below, influence is `1 / d` and the diagonal of `I`
is 0. Because `1/d` decreases as `d` grows, the dominance test
`I(a, c) ≥ I(a, q)` is the same comparison as `d(a, c) ≤ d(a, q)`.

**Canberra**

$$
d_{\mathrm{Can}}(u,v) = \sum_i \frac{|u_i - v_i|}{|u_i| + |v_i|}
$$

A term with `u_i = v_i = 0` is 0.

**Bray-Curtis** (non-negative features)

$$
d_{\mathrm{BC}}(u,v) = \frac{\sum_i |u_i - v_i|}{\sum_i |u_i + v_i|}
$$

A negative entry, or a pair whose denominator is 0, raises `ValueError`.

**Mahalanobis** — `Σ` is the sample covariance of the ranking matrix
(`ddof = 1`). A singular `Σ` raises `ValueError`.

$$
d_M(u,v) = \sqrt{(u-v)^{\mathsf T} \Sigma^{-1} (u-v)}
$$

**Standardized Euclidean** — `V_i` is the sample variance of feature `i`.
A zero variance raises `ValueError`.

$$
d_{\mathrm{SE}}(u,v) = \sqrt{\sum_i (u_i - v_i)^2 / V_i}
$$

**Jensen-Shannon** — each row is turned into a probability `p` (reject
negatives and rows that sum to 0). `m = (p+q)/2`.

$$
d_{\mathrm{JS}}(p,q) = \sqrt{\frac{\mathrm{KL}(p \,\|\, m) + \mathrm{KL}(q \,\|\, m)}{2}}
$$

**Hellinger** — same probabilities, range `[0, 1]`.

$$
d_H(p,q) = \frac{1}{\sqrt{2}} \,\lVert \sqrt{p} - \sqrt{q} \rVert_2
$$

**Spearman** — Pearson correlation of the average ranks of each row.
Rankings use `d = 1 - ρ`. Influence is `ρ` itself. A constant row
raises `ValueError`.

$$
I(u,v) = \mathrm{corr}(\mathrm{rank}(u), \mathrm{rank}(v))
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
│   ├── stages.py          # Anchors, influence, triangulation, mandatory
│   └── io.py              # Directed edge_index, undirected graph
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

Depends on `numpy`, `scipy`, `scikit-learn`, `networkx`.

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
# edge_index: (2, E), int64. Directed list: both directions
# and duplicate pairs may appear.
# G: networkx.Graph, the undirected simple graph on those edges
print(G.number_of_nodes(), G.number_of_edges())
```

If you need to change the knobs:

```python
cfg = BridGConfig(
    k=40,
    mandatory=5,
    rbo_threshold=0.1,
    triangulation=5,
    distance_metric="euclidean",  # see the parameter table
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
| `distance_metric` | `"euclidean"` | `euclidean`, `manhattan`, `chebyshev`, `minkowski`, `cosine`, `correlation`, `canberra`, `braycurtis`, `mahalanobis`, `seuclidean`, `jensenshannon`, `hellinger`, `spearman` |
| `minkowski_p` | `3` | Order $p$ when `distance_metric="minkowski"` (finite, $\geq 1$) |
| `normalize_l2` | `True` | L2 for ranking only |
| `reciprocal` | `True` | Require mutual top-`k` membership |
| `similarity` | `"rbo"` | `"rbo"` or `"jaccard"` |
| `rbo_p` | 0.9 | RBO persistence (truncated residual sum) |
| `anchor_selection` | `"all"` | Dominance anchors: `"all"`, `"random"`, `"probabilistic"`, `"incremental"` |
| `anchor_probability` | `0.5` | Inclusion probability when `anchor_selection="random"` |
| `anchor_seed` | `0` | Seed for `"random"`, `"probabilistic"`, and `"incremental"` draws |

### Tests

```bash
pip install -e ".[dev]"

# Fast suite (skips the N=1000 stress test)
pytest -m "not slow"

# Input-error reports (exception type + message)
pytest -v tests/test_errors.py

# Stress: N=1000 with paper defaults. -s prints the graph summary
pytest -s -v -m slow --durations=10
```

`tests/test_parity.py` imports `brid_g.graphs` and is skipped when
that package is not installed.

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
