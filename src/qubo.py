"""QUBO / Max-Cut formulation of the grid subgraph (Task B).

Turns the weighted subgraph in ``data/grid_cr.json`` into a QUBO (and the
equivalent Ising Hamiltonian) suitable for Max-Cut / QAOA. One binary variable
``x_i in {0, 1}`` is assigned to each node; ``x_i`` labels the fault-zone
partition the substation belongs to.

Objective (minimize the cut by default)::

    CutValue(x) = sum_{(i,j) in E} w_ij * (x_i + x_j - 2 * x_i * x_j)

The term ``x_i + x_j - 2 * x_i * x_j`` is 1 exactly when the edge is cut
(``x_i != x_j``) and 0 otherwise, so ``CutValue`` is the total weight of the cut
lines. QAOA/QUBO *minimize* a cost. By default the cost is
``+CutValue + penalties`` (minimize the cut); set ``maximize_cut=True`` for the
classic ``-CutValue + penalties`` (maximize the cut) instead.

Edge weights are recomputed with the ``generation_inverted`` scheme (the
sign-inverted generation weight) from each edge's stored ``voltage`` and its
endpoints' generators, so the mostly-negative generation weights become mostly
positive with the *most critical* lines scoring *highest*. Minimizing the cut
then makes the fault-zone boundary avoid those critical lines. ``w_max`` (the
largest edge-weight magnitude) is the scale reference for every penalty
coefficient.

Penalties (see ``PENALTIES``), both quadratic (no ancilla qubits):

- ``generator_spread`` -- pairwise same-partition penalty over the generator
  nodes (substations with ``n_generators > 0``). It penalizes every pair of
  generator nodes that lands in the same partition, which is symmetric and so
  discourages both "all generators off" (all-0) and "all generators on one
  side" (all-1), keeping generation on both sides of the cut. Coefficient
  ``P_gen = gen_penalty_factor * w_max`` per generator-node pair.
- ``balance`` -- ``lambda * (sum_i x_i - n/2) ** 2``, which discourages lopsided
  cuts and pushes toward two comparably sized fault zones. Coefficient
  ``lambda = balance_penalty_factor * w_max``.

See ``notebooks/validation.ipynb`` for a step-by-step validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from itertools import combinations
from pathlib import Path

import networkx as nx

from src import weights

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "grid_cr.json"
DEFAULT_OUTPUT = ROOT / "data" / "qubo_cr.json"

# Weight scheme used for the QUBO objective. ``generation_inverted`` makes the
# generation weights mostly positive, with the most critical lines scoring
# highest, for a minimize-cut objective (the boundary avoids critical lines).
QUBO_WEIGHT_SCHEME = "generation_inverted"

# Optimization sense. ``False`` -> minimize the total cut weight, so the
# fault-zone boundary avoids the high-weight (critical) lines. Set ``True`` to
# maximize the cut instead (e.g. with a positive scheme such as ``kv``).
MAXIMIZE_CUT = False

# Default penalty coefficients, expressed as multiples of the maximum edge
# weight magnitude so they stay invariant to the weight scale.
DEFAULT_GEN_PENALTY_FACTOR = 0.5
DEFAULT_BALANCE_PENALTY_FACTOR = 0.15


@dataclass
class QUBO:
    """A QUBO in standard form over binary variables ``x_i in {0, 1}``.

    ``cost(x) = sum_i linear[i] * x_i
              + sum_{i<j} quadratic[(i, j)] * x_i * x_j
              + offset``

    ``variables`` lists the node ids in the fixed order used to index the
    binary variables (0-based); ``index`` is the inverse mapping.
    """

    variables: list[str]
    linear: dict[int, float]
    quadratic: dict[tuple[int, int], float]
    offset: float = 0.0
    metadata: dict = field(default_factory=dict)

    @property
    def index(self) -> dict[str, int]:
        """Map each node id to its variable index."""
        return {name: i for i, name in enumerate(self.variables)}

    def add_linear(self, i: int, value: float) -> None:
        """Accumulate ``value`` on the linear term of variable ``i``."""
        self.linear[i] = self.linear.get(i, 0.0) + value

    def add_quadratic(self, i: int, j: int, value: float) -> None:
        """Accumulate ``value`` on the quadratic term of the pair ``(i, j)``.

        The key is always stored with ``i < j``.
        """
        if i == j:
            self.add_linear(i, value)
            return
        key = (i, j) if i < j else (j, i)
        self.quadratic[key] = self.quadratic.get(key, 0.0) + value

    def energy(self, assignment: dict[str, int] | list[int]) -> float:
        """Evaluate the QUBO cost for a binary assignment.

        ``assignment`` is either a mapping ``node id -> {0, 1}`` or a list of
        bits ordered like ``variables``.
        """
        if isinstance(assignment, dict):
            x = [int(assignment[name]) for name in self.variables]
        else:
            x = [int(b) for b in assignment]
        e = self.offset
        for i, coeff in self.linear.items():
            e += coeff * x[i]
        for (i, j), coeff in self.quadratic.items():
            e += coeff * x[i] * x[j]
        return e

    def to_matrix(self) -> list[list[float]]:
        """Return the symmetric QUBO matrix ``Q`` (linear terms on the diagonal)."""
        n = len(self.variables)
        Q = [[0.0] * n for _ in range(n)]
        for i, coeff in self.linear.items():
            Q[i][i] += coeff
        for (i, j), coeff in self.quadratic.items():
            Q[i][j] += coeff / 2.0
            Q[j][i] += coeff / 2.0
        return Q


# --------------------------------------------------------------------------
# Objective and penalties
# --------------------------------------------------------------------------

def add_maxcut_objective(qubo: QUBO, G: nx.Graph, maximize: bool = True) -> None:
    """Add the cut objective to ``qubo``.

    ``CutValue(x) = sum_{(u,v)} w * (x_u + x_v - 2 x_u x_v)``.

    - ``maximize=True``  -> add ``-CutValue`` (minimizing the cost maximizes the
      cut): per edge ``-w`` on ``x_u, x_v`` and ``+2w`` on ``x_u x_v``.
    - ``maximize=False`` -> add ``+CutValue`` (minimizing the cost minimizes the
      cut, so the boundary avoids high-weight lines): per edge ``+w`` on
      ``x_u, x_v`` and ``-2w`` on ``x_u x_v``.
    """
    s = -1.0 if maximize else 1.0
    idx = qubo.index
    for u, v, d in G.edges(data=True):
        w = float(d.get("weight", 0.0))
        i, j = idx[u], idx[v]
        qubo.add_linear(i, s * w)
        qubo.add_linear(j, s * w)
        qubo.add_quadratic(i, j, -2.0 * s * w)


def add_generator_spread_penalty(qubo: QUBO, G: nx.Graph, coefficient: float) -> None:
    """Penalize generator-node pairs that share a partition (symmetric).

    For every pair ``(i, j)`` of generator nodes the same-partition indicator is
    ``1 - (x_i + x_j - 2 x_i x_j) = 1 - x_i - x_j + 2 x_i x_j`` (1 when
    ``x_i == x_j``). Summed over pairs and scaled by ``coefficient``::

        P * sum_{gen pairs} (1 - x_i - x_j + 2 x_i x_j)

    which is minimized when the generators are spread across both partitions.
    """
    if coefficient == 0:
        return
    idx = qubo.index
    gen_nodes = generator_nodes(G)
    for u, v in combinations(gen_nodes, 2):
        i, j = idx[u], idx[v]
        qubo.offset += coefficient
        qubo.add_linear(i, -coefficient)
        qubo.add_linear(j, -coefficient)
        qubo.add_quadratic(i, j, 2.0 * coefficient)


def add_balance_penalty(qubo: QUBO, G: nx.Graph, coefficient: float) -> None:
    """Penalize lopsided partitions via ``lambda * (sum_i x_i - n/2) ** 2``.

    Expanding with ``x_i ** 2 = x_i``::

        (sum x_i - n/2) ** 2 = (1 - n) * sum x_i
                             + 2 * sum_{i<j} x_i x_j
                             + n**2 / 4
    """
    if coefficient == 0:
        return
    n = len(qubo.variables)
    qubo.offset += coefficient * (n ** 2) / 4.0
    for i in range(n):
        qubo.add_linear(i, coefficient * (1 - n))
    for i, j in combinations(range(n), 2):
        qubo.add_quadratic(i, j, 2.0 * coefficient)


# Registry of penalty builders. Each entry is
# ``fn(qubo, G, coefficient) -> None`` and mutates ``qubo`` in place, mirroring
# the ``weights.SCHEMES`` registry convention. Add a penalty by adding a
# function and an entry here -- do not hardcode penalties elsewhere.
PENALTIES = {
    "generator_spread": add_generator_spread_penalty,
    "balance": add_balance_penalty,
}


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def generator_nodes(G: nx.Graph) -> list[str]:
    """Return the ids of nodes with at least one associated generator.

    Sorted alphabetically for deterministic penalty construction.
    """
    return sorted(
        n for n, d in G.nodes(data=True) if int(d.get("n_generators", 0) or 0) > 0
    )


def max_edge_weight(G: nx.Graph) -> float:
    """Return the weight scale ``w_max`` = largest edge-weight *magnitude*.

    Using the absolute value keeps penalty coefficients well-scaled even when
    the objective weights are mixed-sign (as with ``generation_inverted``).
    Returns 1.0 if there are no edges.
    """
    ws = [abs(float(d.get("weight", 0.0))) for _, _, d in G.edges(data=True)]
    return max(ws) if ws else 1.0


def apply_weight_scheme(G: nx.Graph, weight_scheme: str) -> nx.Graph:
    """Return a copy of ``G`` with edge weights recomputed from the scheme.

    Uses ``weights.SCHEMES[weight_scheme]``, passing each endpoint's generator
    list so generator-aware schemes (``generation`` / ``generation_inverted``)
    reproduce their full context; voltage-only schemes (the ``kv`` family)
    ignore it.
    """
    weight_fn = weights.SCHEMES[weight_scheme]
    H = G.copy()
    for u, v, d in H.edges(data=True):
        d["weight"] = weight_fn(
            voltage=d.get("voltage"),
            length_m=d.get("length_m"),
            gens_u=H.nodes[u].get("generators", []),
            gens_v=H.nodes[v].get("generators", []),
        )
    return H


def build_qubo(
    G: nx.Graph,
    weight_scheme: str = QUBO_WEIGHT_SCHEME,
    maximize_cut: bool = MAXIMIZE_CUT,
    gen_penalty_factor: float = DEFAULT_GEN_PENALTY_FACTOR,
    balance_penalty_factor: float = DEFAULT_BALANCE_PENALTY_FACTOR,
) -> QUBO:
    """Build the cut QUBO for the graph ``G``.

    Edge weights are recomputed with ``weight_scheme`` (default
    ``generation_inverted``: critical lines score highest and positive). With
    ``maximize_cut=False`` (default) the objective *minimizes* the cut weight, so
    the fault-zone boundary avoids the high-weight (critical) lines; set
    ``maximize_cut=True`` for the classic max-cut sense.

    Penalty coefficients are expressed as multiples of the weight scale ``w_max``
    (largest edge-weight magnitude) so they stay invariant to the weight scale::

        P_gen  = gen_penalty_factor     * w_max   (per generator-node pair)
        lambda = balance_penalty_factor * w_max

    Node variables are indexed in sorted order for reproducibility.
    """
    H = apply_weight_scheme(G, weight_scheme)
    variables = sorted(H.nodes)
    qubo = QUBO(variables=variables, linear={}, quadratic={})

    w_max = max_edge_weight(H)
    gen_coeff = gen_penalty_factor * w_max
    balance_coeff = balance_penalty_factor * w_max

    add_maxcut_objective(qubo, H, maximize=maximize_cut)
    PENALTIES["generator_spread"](qubo, H, gen_coeff)
    PENALTIES["balance"](qubo, H, balance_coeff)

    gens = generator_nodes(H)
    qubo.metadata = {
        "weight_scheme": weight_scheme,
        "maximize_cut": maximize_cut,
        "n_variables": len(variables),
        "n_edges": H.number_of_edges(),
        "max_edge_weight": w_max,
        "generator_nodes": gens,
        "n_generator_nodes": len(gens),
        "n_generator_pairs": len(gens) * (len(gens) - 1) // 2,
        "penalties": {
            "generator_spread": {
                "factor": gen_penalty_factor,
                "coefficient": gen_coeff,
            },
            "balance": {
                "factor": balance_penalty_factor,
                "coefficient": balance_coeff,
            },
        },
    }
    return qubo


# --------------------------------------------------------------------------
# Ising conversion
# --------------------------------------------------------------------------

def qubo_to_ising(qubo: QUBO) -> dict:
    """Convert the QUBO to an Ising Hamiltonian for QAOA.

    Uses ``x_i = (1 - z_i) / 2`` with ``z_i in {-1, +1}``. Returns a dict with:

    - ``h``: list of local fields ``h_i`` (one per variable);
    - ``J``: mapping ``(i, j) -> J_ij`` couplings (``i < j``);
    - ``offset``: constant energy shift.

    such that ``cost(x) = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + offset``.
    """
    n = len(qubo.variables)
    h = [0.0] * n
    J: dict[tuple[int, int], float] = {}
    offset = qubo.offset

    # Linear:  c * x_i = c * (1 - z_i)/2 = c/2 - (c/2) z_i
    for i, c in qubo.linear.items():
        offset += c / 2.0
        h[i] += -c / 2.0

    # Quadratic: c * x_i x_j = c/4 * (1 - z_i)(1 - z_j)
    #          = c/4 - (c/4) z_i - (c/4) z_j + (c/4) z_i z_j
    for (i, j), c in qubo.quadratic.items():
        offset += c / 4.0
        h[i] += -c / 4.0
        h[j] += -c / 4.0
        J[(i, j)] = J.get((i, j), 0.0) + c / 4.0

    return {"h": h, "J": J, "offset": offset}


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------

def load_graph(path: Path = DEFAULT_INPUT) -> nx.Graph:
    """Load a subgraph from a ``grid_cr.json`` document into a NetworkX graph."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    G = nx.Graph()
    for node in doc.get("nodes", []):
        attrs = {k: v for k, v in node.items() if k != "id"}
        G.add_node(node["id"], **attrs)
    for edge in doc.get("edges", []):
        attrs = {k: v for k, v in edge.items() if k not in ("u", "v")}
        G.add_edge(edge["u"], edge["v"], **attrs)
    return G


def to_json(qubo: QUBO, metadata: dict | None = None) -> dict:
    """Serialize a QUBO (and its Ising form) to a JSON-ready dict.

    Quadratic keys are stringified as ``"i,j"`` for JSON compatibility.
    """
    ising = qubo_to_ising(qubo)
    return {
        "metadata": {**qubo.metadata, **(metadata or {})},
        "variables": qubo.variables,
        "qubo": {
            "linear": {str(i): c for i, c in sorted(qubo.linear.items())},
            "quadratic": {
                f"{i},{j}": c for (i, j), c in sorted(qubo.quadratic.items())
            },
            "offset": qubo.offset,
        },
        "ising": {
            "h": ising["h"],
            "J": {f"{i},{j}": c for (i, j), c in sorted(ising["J"].items())},
            "offset": ising["offset"],
        },
    }


def save_qubo(qubo: QUBO, path: Path, metadata: dict | None = None) -> Path:
    """Write the QUBO document to disk in JSON format."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = to_json(qubo, metadata)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build(
    input_path: Path = DEFAULT_INPUT,
    output: Path = DEFAULT_OUTPUT,
    weight_scheme: str = QUBO_WEIGHT_SCHEME,
    maximize_cut: bool = MAXIMIZE_CUT,
    gen_penalty_factor: float = DEFAULT_GEN_PENALTY_FACTOR,
    balance_penalty_factor: float = DEFAULT_BALANCE_PENALTY_FACTOR,
) -> QUBO:
    """Read ``grid_cr.json``, build the QUBO, and write ``qubo_cr.json``."""
    doc = json.loads(Path(input_path).read_text(encoding="utf-8"))
    G = load_graph(input_path)
    qubo = build_qubo(
        G,
        weight_scheme=weight_scheme,
        maximize_cut=maximize_cut,
        gen_penalty_factor=gen_penalty_factor,
        balance_penalty_factor=balance_penalty_factor,
    )
    metadata = {
        "build_date": date.today().isoformat(),
        "source_graph": str(Path(input_path).name),
        "region": doc.get("metadata", {}).get("region"),
    }
    save_qubo(qubo, output, metadata)
    return qubo


if __name__ == "__main__":
    q = build()
    print(f"QUBO: {q.metadata['n_variables']} variables, "
          f"{len(q.quadratic)} quadratic terms -> {DEFAULT_OUTPUT}")
    print(f"Generator nodes ({q.metadata['n_generator_nodes']}): "
          f"{', '.join(q.metadata['generator_nodes'])}")
