"""Descarga y snapshot de los datos abiertos del ICE (portal ArcGIS).

Este módulo obtiene las dos capas que necesitamos para construir el grafo de la
red eléctrica y guarda una copia estática (snapshot) en ``data/raw/``. El resto
del pipeline lee siempre desde el snapshot, nunca en vivo, para garantizar la
reproducibilidad de la entrega.

Capas:
  - Subestaciones      -> nodos del grafo
  - LineasDeTransmision -> aristas del grafo

Fuente: Portal de Datos Abiertos del Sector Electricidad del ICE (ArcGIS Hub).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

# Servicio de features (ArcGIS REST) del ICE.
BASE_URL = "https://services1.arcgis.com/cW2GfO4rBCLoFwgj/arcgis/rest/services"

# Cada entrada: nombre de la capa -> (ruta del servicio, archivo de snapshot).
LAYERS = {
    "subestaciones": "Subestaciones/FeatureServer/0",
    "lineas": "LineasDeTransmision/FeatureServer/0",
}

# Directorio donde guardamos el snapshot estático (relativo a la raíz del repo).
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def _query_url(layer_path: str) -> str:
    """Construye la URL de consulta GeoJSON para descargar la capa completa."""
    params = urllib.parse.urlencode(
        {"where": "1=1", "outFields": "*", "f": "geojson"}
    )
    return f"{BASE_URL}/{layer_path}/query?{params}"


def download_layer(layer_path: str, timeout: int = 60) -> dict:
    """Descarga una capa del servicio ArcGIS y la devuelve como dict GeoJSON."""
    url = _query_url(layer_path)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def snapshot(force: bool = False, raw_dir: Path = RAW_DIR) -> dict[str, Path]:
    """Descarga las capas y las guarda en ``data/raw/``.

    Si el snapshot ya existe no se vuelve a descargar, salvo ``force=True``.
    Devuelve un mapa nombre_de_capa -> ruta del archivo guardado.
    Además escribe ``data/raw/fuente.json`` con las URLs y la fecha de descarga.
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

    # Registro de procedencia para el informe y la reproducibilidad.
    (raw_dir / "fuente.json").write_text(
        json.dumps(
            {
                "fuente": "Portal de Datos Abiertos del Sector Electricidad del ICE (ArcGIS Hub)",
                "urls": urls,
                "fecha_descarga": date.today().isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return paths


def load_snapshot(name: str, raw_dir: Path = RAW_DIR) -> dict:
    """Carga una capa desde el snapshot local (GeoJSON)."""
    path = raw_dir / f"{name}.geojson"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe el snapshot '{path}'. Ejecuta ice_data.snapshot() primero."
        )
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    saved = snapshot()
    for name, path in saved.items():
        data = load_snapshot(name)
        n = len(data.get("features", []))
        print(f"{name}: {n} features -> {path}")
