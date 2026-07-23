from __future__ import annotations

from src import classical_baselines as cb
from src import qubo


def test_project_graph_baselines_run_on_repo_data() -> None:
    G = qubo.load_graph()

    partition, value = cb.greedy_maxcut(G, seed=0)

    assert set(partition) == set(G.nodes)
    assert isinstance(value, float)
    assert value == cb.maxcut_value(G, partition)

    bf_partition, bf_value = cb.brute_force_maxcut(G)
    assert set(bf_partition) == set(G.nodes)
    assert bf_value == cb.maxcut_value(G, bf_partition)
