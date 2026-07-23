from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src import classical_baselines as cb
from src import qubo


def test_classical_baselines_script_runs_directly() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "src" / "classical_baselines.py"

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_project_graph_baselines_run_on_repo_data() -> None:
    G = qubo.load_graph()

    partition, value = cb.greedy_maxcut(G, seed=0)

    assert set(partition) == set(G.nodes)
    assert isinstance(value, float)
    assert value == cb.maxcut_value(G, partition)

    bf_partition, bf_value = cb.brute_force_maxcut(G)
    assert set(bf_partition) == set(G.nodes)
    assert bf_value == cb.maxcut_value(G, bf_partition)
