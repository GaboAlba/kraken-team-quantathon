"""Download and snapshot of the ICE open data (ArcGIS portal).

This module fetches the two layers we need to build the electrical grid graph
and stores a static copy (snapshot) in ``data/raw/``. The rest of the pipeline
always reads from the snapshot, never live, to guarantee the reproducibility of
the deliverable.

Layers:
  - Subestaciones       -> graph nodes (substations)
  - LineasDeTransmision -> graph edges (transmission lines)
  - Plantas_NGICE       -> generation plants (node annotation: technology, MW)

Source: ICE Electricity Sector Open Data Portal (ArcGIS Hub).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

# ICE feature service (ArcGIS REST).
BASE_URL = "https://services1.arcgis.com/cW2GfO4rBCLoFwgj/arcgis/rest/services"

# Each entry: layer name -> service path (snapshot file is "<name>.geojson").
LAYERS = {
    "substations": "Subestaciones/FeatureServer/0",
    "lines": "LineasDeTransmision/FeatureServer/0",
    "plants": "Plantas_NGICE/FeatureServer/0",
}

# Directory where the static snapshot is stored (relative to the repo root).
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _query_url(layer_path: str) -> str:
    """Build the GeoJSON query URL to download the full layer."""
    params = urllib.parse.urlencode(
        {"where": "1=1", "outFields": "*", "f": "geojson"}
    )
    return f"{BASE_URL}/{layer_path}/query?{params}"


def download_layer(layer_path: str, timeout: int = 60) -> dict:
    """Download a layer from the ArcGIS service and return it as a GeoJSON dict."""
    url = _query_url(layer_path)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def snapshot(force: bool = False, raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    """Download the layers and store them in ``data/raw/``.

    If the snapshot already exists it is not downloaded again, unless
    ``force=True``. Returns a map layer_name -> path of the saved file.
    Also writes ``data/raw/source.json`` with the URLs and the download date.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    urls: dict[str, str] = {}

    for name, layer_path in LAYERS.items():
        out = raw_dir / f"{name}.geojson"
        urls[name] = _query_url(layer_path)
        if out.exists() and not force:
            paths[name] = out
            continue
        data = download_layer(layer_path)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        paths[name] = out

    # Provenance record for the report and reproducibility.
    (raw_dir / "source.json").write_text(
        json.dumps(
            {
                "source": "ICE Electricity Sector Open Data Portal (ArcGIS Hub)",
                "urls": urls,
                "download_date": date.today().isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def load_snapshot(name: str, raw_dir: Path = RAW_DIR) -> dict:
    """Load a layer from the local snapshot (GeoJSON)."""
    path = raw_dir / f"{name}.geojson"
    if not path.exists():
        raise FileNotFoundError(
            f"Snapshot '{path}' does not exist. Run ice_data.snapshot() first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    saved = snapshot()
    for name, path in saved.items():
        data = load_snapshot(name)
        n = len(data.get("features", []))
        print(f"{name}: {n} features -> {path}")
