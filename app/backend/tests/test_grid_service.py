import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grid_service


def test_grid_payload_shapes():
    doc = grid_service.grid_payload()
    assert {"nodes", "edges", "plants"} <= set(doc)
    node = doc["nodes"][0]
    assert {"id", "name", "lat", "lon", "is_initial"} <= set(node)
    assert any(n["is_initial"] for n in doc["nodes"])
    edge = doc["edges"][0]
    assert {"u", "v", "voltage", "weight"} <= set(edge)
    plant = doc["plants"][0]
    assert {"name", "technology", "mw", "lat", "lon", "substation"} <= set(plant)
    # At least the co-located plants resolve to a substation.
    assert any(p["substation"] for p in doc["plants"])


def test_initial_subgrid_is_valid():
    info = grid_service.subgrid_info(list(grid_service.INITIAL_NODES))
    assert info["valid"] is True
    assert len(info["nodes"]) == 9
    assert len(info["edges"]) == 9
    assert "colorado" in info["adjacent"]      # neighbor of canas


def test_subgrid_missing_initial_node_is_invalid():
    nodes = [n for n in grid_service.INITIAL_NODES if n != "arenal"]
    info = grid_service.subgrid_info(nodes)
    assert info["valid"] is False
    assert "arenal" in info["reason"]


def test_subgrid_disconnected_is_invalid():
    # cobano is far from the northern ring -> induced graph disconnects.
    info = grid_service.subgrid_info(list(grid_service.INITIAL_NODES) + ["cobano"])
    assert info["valid"] is False
    assert "connect" in info["reason"].lower()


def _grow_selection(target: int) -> list[str]:
    """Grow a connected selection from the initial nodes via adjacency."""
    selection = list(grid_service.INITIAL_NODES)
    while len(selection) < target:
        info = grid_service.subgrid_info(selection)
        assert info["adjacent"], f"no candidates left at {len(selection)}"
        selection.append(info["adjacent"][0])
    return selection


def test_subgrid_tiers_by_size():
    # exact: full exact pipeline; heuristic: brute force still RUNS but QAOA
    # angles are untuned; classical: brute force physically impossible.
    assert grid_service.tier_for(9) == "exact"
    assert grid_service.tier_for(22) == "exact"
    assert grid_service.tier_for(23) == "heuristic"
    assert grid_service.tier_for(40) == "heuristic"
    assert grid_service.tier_for(41) == "classical"


def test_large_selections_are_valid_with_tier():
    sel16 = _grow_selection(16)
    info = grid_service.subgrid_info(sel16)
    assert info["valid"] is True
    assert info["tier"] == "exact"

    sel25 = _grow_selection(25)
    info = grid_service.subgrid_info(sel25)
    assert info["valid"] is True
    assert info["tier"] == "heuristic"

    sel45 = _grow_selection(45)
    info = grid_service.subgrid_info(sel45)
    assert info["valid"] is True
    assert info["tier"] == "classical"
