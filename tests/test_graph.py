"""Tests for the graph construction module (Task A).

They combine small synthetic cases (for pure, deterministic logic) with checks
against the real ICE snapshot in ``data/raw/``.
"""

import sys
from pathlib import Path

import networkx as nx
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import graph
from src import weights


# --------------------------------------------------------------------------
# Name normalization
# --------------------------------------------------------------------------

def test_normalize_strips_accents_and_lowercases():
    assert graph.normalize_name("Liberón") == "liberon"
    assert graph.normalize_name("  Río  Macho ") == "rio macho"


def test_normalize_drops_trailing_bay_digit():
    # Numbered circuits (colima1, colima2) point to the same substation.
    assert graph.normalize_name("Colima1") == graph.normalize_name("Colima2")
    assert graph.normalize_name("Colima2") == "colima"


def test_normalize_drops_siepac_suffix():
    assert graph.normalize_name("Jaco (SIEPAC)") == "jaco"


# --------------------------------------------------------------------------
# Circuit parsing
# --------------------------------------------------------------------------

def test_parse_circuit_two_endpoints():
    assert graph.parse_circuit("Liberia-Papagayo") == ("liberia", "papagayo")


def test_parse_circuit_invalid_format_returns_none():
    assert graph.parse_circuit("SIEPAC") is None
    assert graph.parse_circuit("") is None
    assert graph.parse_circuit(None) is None


# --------------------------------------------------------------------------
# Weight functions
# --------------------------------------------------------------------------

def test_weight_kv_is_the_voltage():
    assert weights.SCHEMES["kv"](voltage=230, length_m=1000) == 230
    assert weights.SCHEMES["kv"](voltage=138, length_m=5000) == 138


def test_weight_kv_normalized_between_0_and_1():
    w = weights.SCHEMES["kv_normalized"](voltage=230, length_m=1000)
    assert w == pytest.approx(1.0)
    assert weights.SCHEMES["kv_normalized"](voltage=138, length_m=1) == pytest.approx(138 / 230)


def test_weight_all_schemes_are_positive():
    # generation_inverted is intentionally sign-inverted (mostly positive but
    # not guaranteed), so it is excluded from this positivity invariant.
    for name, fn in weights.SCHEMES.items():
        if name == "generation_inverted":
            continue
        assert fn(voltage=138, length_m=2500) > 0, name


def test_generation_weight_base_without_generators():
    fn = weights.SCHEMES["generation"]
    assert fn(voltage=230, length_m=0) == pytest.approx(1 - 0.23)


def test_generation_weight_counts_generators_and_both_ends_penalty():
    fn = weights.SCHEMES["generation"]
    gens_u = [{"thermal": False}]
    gens_v = [{"thermal": False}, {"thermal": False}]
    expected = 1 - 1 - 2 + 0.5 - 230 / 1000
    assert fn(voltage=230, length_m=0, gens_u=gens_u, gens_v=gens_v) == pytest.approx(expected)


def test_generation_weight_does_not_apply_both_ends_penalty_to_one_side():
    fn = weights.SCHEMES["generation"]
    gens_u = [{"thermal": False}]
    expected = 1 - 1 - 230 / 1000
    assert fn(voltage=230, length_m=0, gens_u=gens_u, gens_v=[]) == pytest.approx(expected)


def test_generation_weight_applies_halving_thermal_penalty():
    fn = weights.SCHEMES["generation"]
    gens_u = [{"thermal": True}, {"thermal": True}]
    gens_v = [{"thermal": True}]
    expected = 1 - 2 - 1 + 0.5 + (0.5 + 0.25 + 0.125) - 230 / 1000
    assert fn(voltage=230, length_m=0, gens_u=gens_u, gens_v=gens_v) == pytest.approx(expected)


def test_generation_weight_adds_normalized_generator_power():
    fn = weights.SCHEMES["generation"]
    # Important lines are cheaper: the biggest generator (power_norm == 1.0)
    # lowers the weight by a full point; a half-size one by 0.5. Non-thermal.
    gens_u = [{"thermal": False, "power_norm": 1.0}]
    gens_v = [{"thermal": False, "power_norm": 0.5}]
    expected = 1 - 1 - 1 - (1.0 + 0.5) + 0.5 - 230 / 1000
    assert fn(voltage=230, length_m=0, gens_u=gens_u, gens_v=gens_v) == pytest.approx(expected)


# --------------------------------------------------------------------------
# Graph construction with synthetic data
# --------------------------------------------------------------------------

def _fake_geojson_subs(names_provinces):
    feats = []
    for i, (name, prov) in enumerate(names_provinces):
        feats.append({
            "properties": {"Subestacio": name, "Provincia": prov, "Canton": "X"},
            "geometry": {"type": "Point", "coordinates": [float(i), float(i)]},
        })
    return {"features": feats}


def _fake_geojson_lines(circuits_voltages):
    feats = []
    for circ, volt in circuits_voltages:
        feats.append({
            "properties": {"Circuito": circ, "Voltaje": volt, "Shape__Length": 1000.0},
            "geometry": {"type": "MultiLineString", "coordinates": [[[0, 0], [1, 1]]]},
        })
    return {"features": feats}


def _fake_geojson_plants(plants):
    feats = []
    for plant, technology, status, x, y, power_mw in plants:
        feats.append({
            "properties": {
                "Planta": plant,
                "Tecnologia": technology,
                "EstAct": status,
                "XCoord": x,
                "YCoord": y,
                "PotenciaEfectivaMW": power_mw,
            },
            "geometry": {"type": "Point", "coordinates": [x, y]},
        })
    return {"features": feats}


def test_is_thermal_handles_accents_case_and_non_thermal_technologies():
    for technology in ("Térmico", "Térmica", "térmico"):
        assert graph.is_thermal(technology) is True
    for technology in ("Geotérmico", "Hidroeléctrico", "Solar", None, ""):
        assert graph.is_thermal(technology) is False


def test_parse_generators_keeps_active_plants_and_projected_coordinates():
    plants = _fake_geojson_plants([
        ("Thermal Plant", "Térmico", "Activo", 100.0, 200.0, 10.5),
        ("Inactive Plant", "Solar", "Inactivo", 300.0, 400.0, 20.0),
        ("Hydro Plant", "Hidroeléctrico", "Activo", 500.0, 600.0, 30.0),
    ])
    generators = graph.parse_generators(plants)
    assert generators == [
        {
            "plant": "Thermal Plant",
            "technology": "Térmico",
            "thermal": True,
            "power_mw": 10.5,
            "x": 100.0,
            "y": 200.0,
        },
        {
            "plant": "Hydro Plant",
            "technology": "Hidroeléctrico",
            "thermal": False,
            "power_mw": 30.0,
            "x": 500.0,
            "y": 600.0,
        },
    ]


def test_distance_m_uses_euclidean_distance():
    assert graph._distance_m(0, 0, 3, 4) == pytest.approx(5)


def test_assign_generators_respects_radius_and_sorts_by_distance():
    G = nx.Graph()
    G.add_node("substation", px=0.0, py=0.0, border=False)
    G.add_node("border", px=None, py=None, border=True)
    G.add_node("no_coord", px=None, py=0.0, border=False)
    generators = [
        {"plant": "Far Plant", "technology": "Solar", "thermal": False,
         "power_mw": 3.0, "x": 25000.0, "y": 0.0},
        {"plant": "Plant B", "technology": "Solar", "thermal": False,
         "power_mw": 2.0, "x": 1000.0, "y": 0.0},
        {"plant": "Plant A", "technology": "Térmico", "thermal": True,
         "power_mw": 1.0, "x": 500.0, "y": 0.0},
    ]

    graph.assign_generators(G, generators, radius_m=20000.0)

    assert G.nodes["substation"]["n_generators"] == 2
    assert G.nodes["substation"]["n_thermal"] == 1
    assert [g["plant"] for g in G.nodes["substation"]["generators"]] == ["Plant A", "Plant B"]
    assert [g["dist_m"] for g in G.nodes["substation"]["generators"]] == [500.0, 1000.0]
    # Far Plant (3 MW) is out of radius, so the biggest attached generator is
    # Plant B (2 MW): it scores 1.0 and Plant A (1 MW) scores 0.5.
    assert G.graph["max_generator_power_mw"] == 2.0
    norms = {g["plant"]: g["power_norm"] for g in G.nodes["substation"]["generators"]}
    assert norms == {"Plant A": pytest.approx(0.5), "Plant B": pytest.approx(1.0)}
    assert G.nodes["border"]["generators"] == []
    assert G.nodes["border"]["n_generators"] == 0
    assert G.nodes["no_coord"]["generators"] == []


def test_build_national_graph_basic():
    subs = _fake_geojson_subs([("Liberia", "Guanacaste"), ("Papagayo", "Guanacaste"),
                               ("Canas", "Guanacaste")])
    lines = _fake_geojson_lines([("Liberia-Papagayo", 230), ("Canas-Liberia", 138)])
    G, report = graph.build_national_graph(subs, lines, weight_scheme="kv")
    assert set(G.nodes) == {"liberia", "papagayo", "canas"}
    assert G.number_of_edges() == 2
    assert G["liberia"]["papagayo"]["weight"] == 230
    assert G.nodes["liberia"]["province"] == "Guanacaste"


def test_parallel_lines_are_collapsed_summing_weight():
    subs = _fake_geojson_subs([("Liberia", "Guanacaste"), ("Papagayo", "Guanacaste")])
    lines = _fake_geojson_lines([("Liberia-Papagayo", 230), ("Liberia-Papagayo", 230)])
    G, _ = graph.build_national_graph(subs, lines, weight_scheme="kv")
    assert G.number_of_edges() == 1
    assert G["liberia"]["papagayo"]["weight"] == 460


def test_build_national_graph_with_plants_uses_generation_weight():
    subs = {
        "features": [
            {
                "properties": {
                    "Subestacio": "Alpha",
                    "Provincia": "G",
                    "Canton": "X",
                    "PuntoX": 0.0,
                    "PuntoY": 0.0,
                },
                "geometry": {"type": "Point", "coordinates": [0.0, 0.0]},
            },
            {
                "properties": {
                    "Subestacio": "Beta",
                    "Provincia": "G",
                    "Canton": "X",
                    "PuntoX": 30000.0,
                    "PuntoY": 0.0,
                },
                "geometry": {"type": "Point", "coordinates": [1.0, 1.0]},
            },
        ]
    }
    lines = _fake_geojson_lines([("Alpha-Beta", 230)])
    plants = _fake_geojson_plants([
        ("Hydro Near Alpha", "Hidroeléctrico", "Activo", 1000.0, 0.0, 5.0),
        ("Thermal Near Beta", "Térmico", "Activo", 31000.0, 0.0, 6.0),
        ("Far Solar", "Solar", "Activo", 51000.0, 0.0, 7.0),
    ])

    G, _ = graph.build_national_graph(subs, lines, plants_geojson=plants,
                                      weight_scheme="generation")

    assert G.nodes["alpha"]["n_generators"] == 1
    assert G.nodes["beta"]["n_generators"] == 1
    assert G.nodes["beta"]["n_thermal"] == 1
    # Biggest attached generator is the 6 MW thermal one (Far Solar is out of
    # radius), so it scores 1.0 and the 5 MW hydro scores 5/6.
    assert G.graph["max_generator_power_mw"] == 6.0
    expected = 1 - 1 - 1 - (5 / 6 + 1.0) + 0.5 + 0.5 - 230 / 1000
    assert G["alpha"]["beta"]["weight"] == pytest.approx(expected)
    doc = graph.to_json(G)
    node = next(n for n in doc["nodes"] if n["id"] == "alpha")
    assert {"generators", "n_generators", "n_thermal"} <= set(node)


def test_unrecognized_endpoint_is_marked_as_border():
    subs = _fake_geojson_subs([("Liberia", "Guanacaste")])
    lines = _fake_geojson_lines([("Liberia-Frontera Nicaragua", 230)])
    G, report = graph.build_national_graph(subs, lines)
    assert G.nodes["liberia"]["border"] is False
    assert G.nodes["frontera nicaragua"]["border"] is True
    assert "frontera nicaragua" in report["unrecognized_endpoints"]


# --------------------------------------------------------------------------
# Subregion extraction
# --------------------------------------------------------------------------

def test_extract_subregion_by_province():
    subs = _fake_geojson_subs([("Liberia", "Guanacaste"), ("Papagayo", "Guanacaste"),
                               ("Colima", "Alajuela")])
    lines = _fake_geojson_lines([("Liberia-Papagayo", 230), ("Papagayo-Colima", 138)])
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, region="Guanacaste")
    assert set(sub.nodes) == {"liberia", "papagayo"}


def test_extract_subregion_keeps_largest_connected_component():
    subs = _fake_geojson_subs([("A", "G"), ("B", "G"), ("C", "G"), ("D", "G")])
    # A-B-C connected; D isolated (no edges within the region).
    lines = _fake_geojson_lines([("A-B", 230), ("B-C", 230)])
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, region="G", max_nodes=12)
    assert "d" not in sub.nodes
    assert set(sub.nodes) == {"a", "b", "c"}


def test_extract_subregion_respects_max_nodes():
    subs = _fake_geojson_subs([(c, "G") for c in "ABCDE"])
    lines = _fake_geojson_lines([("A-B", 230), ("B-C", 230), ("C-D", 230), ("D-E", 230)])
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, region="G", max_nodes=3)
    assert sub.number_of_nodes() == 3


# --------------------------------------------------------------------------
# Serialization
# --------------------------------------------------------------------------

def test_to_json_structure(tmp_path):
    subs = _fake_geojson_subs([("Liberia", "Guanacaste"), ("Papagayo", "Guanacaste")])
    lines = _fake_geojson_lines([("Liberia-Papagayo", 230)])
    G, _ = graph.build_national_graph(subs, lines)
    doc = graph.to_json(G, metadata={"region": "Guanacaste", "weight_scheme": "kv"})
    assert doc["metadata"]["region"] == "Guanacaste"
    assert len(doc["nodes"]) == 2
    assert len(doc["edges"]) == 1
    n = doc["nodes"][0]
    assert {
        "id", "name", "province", "canton", "x", "y", "border",
        "generators", "n_generators", "n_thermal",
    } <= set(n)
    e = doc["edges"][0]
    assert {"u", "v", "weight", "voltage", "circuit"} <= set(e)


# --------------------------------------------------------------------------
# Checks against the real ICE snapshot
# --------------------------------------------------------------------------

RAW = ROOT / "data" / "raw"
has_snapshot = (RAW / "substations.geojson").exists() and (RAW / "lines.geojson").exists()
has_plants = has_snapshot and (RAW / "plants.geojson").exists()
real = pytest.mark.skipif(not has_snapshot, reason="ICE snapshot not available")
real_with_plants = pytest.mark.skipif(not has_plants, reason="ICE plants snapshot not available")


@real
def test_real_snapshot_national_graph():
    import json
    subs = json.loads((RAW / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lines.geojson").read_text(encoding="utf-8"))
    G, report = graph.build_national_graph(subs, lines)
    # It must include at least the 70 real substations as nodes.
    real_nodes = [n for n, d in G.nodes(data=True) if not d["border"]]
    assert len(real_nodes) >= 70
    assert G.number_of_edges() > 60


@real
def test_real_snapshot_guanacaste_subgrid_is_valid():
    import json
    subs = json.loads((RAW / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lines.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, region="Guanacaste", max_nodes=12)
    assert 6 <= sub.number_of_nodes() <= 12
    assert nx.is_connected(sub)
    assert sub.number_of_edges() >= sub.number_of_nodes() - 1


@real_with_plants
def test_real_snapshot_with_plants_assigns_generators():
    import json
    subs = json.loads((RAW / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lines.geojson").read_text(encoding="utf-8"))
    plants = json.loads((RAW / "plants.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines, plants_geojson=plants)
    real_nodes = [d for _, d in G.nodes(data=True) if not d["border"]]
    assert any(d["n_generators"] > 0 for d in real_nodes)


@real
def test_real_snapshot_connectivity_mode_preserves_cycles():
    # The default (connectivity) must yield non-trivial instances: with >=8
    # nodes the Max-Cut stops being a tree (at least one independent cycle).
    import json
    subs = json.loads((RAW / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lines.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines)
    for n in (8, 10, 12):
        sub = graph.extract_subregion(G, region=None, max_nodes=n)
        assert nx.is_connected(sub)
        cycles = sub.number_of_edges() - sub.number_of_nodes() + 1
        assert cycles >= 1, f"N={n} came out without cycles"


GUANACASTE_NORTH_EXPECTED = {
    "arenal", "canas", "corobici", "liberia", "miravalles",
    "mogote", "nuevo colon", "pailas", "papagayo",
}


@real
def test_real_snapshot_guanacaste_north_selection():
    import json
    subs = json.loads((RAW / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lines.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines)
    sub = graph.extract_subregion(G, nodes=graph.GUANACASTE_NORTH)
    assert set(sub.nodes) == GUANACASTE_NORTH_EXPECTED
    assert nx.is_connected(sub)
    # The ring Liberia-Pailas-Mogote-Miravalles-Arenal-Corobici-Canas-Liberia.
    assert sub.number_of_edges() - sub.number_of_nodes() + 1 == 1


@real
def test_real_snapshot_build_writes_guanacaste_north(tmp_path):
    out = tmp_path / "grid.json"
    sub = graph.build(nodes=graph.GUANACASTE_NORTH, label="guanacaste_north",
                      max_nodes=len(graph.GUANACASTE_NORTH), output=out)
    import json
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["metadata"]["region"] == "guanacaste_north"
    assert doc["metadata"]["selection_mode"] == "nodes"
    assert {n["id"] for n in doc["nodes"]} == GUANACASTE_NORTH_EXPECTED
    assert sub.number_of_nodes() == 9


@real
def test_real_snapshot_connectivity_is_deterministic():
    import json
    subs = json.loads((RAW / "substations.geojson").read_text(encoding="utf-8"))
    lines = json.loads((RAW / "lines.geojson").read_text(encoding="utf-8"))
    G, _ = graph.build_national_graph(subs, lines)
    a = graph.extract_subregion(G, region=None, max_nodes=10)
    b = graph.extract_subregion(G, region=None, max_nodes=10)
    assert set(a.nodes) == set(b.nodes)
