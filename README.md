# Kraken Team Quantathon Project

## Team Members

* Gabriel Alba Romero
* Juan C Lara

## Challenge
We are solving Challenge #1

## Setup

This project uses a Python virtual environment (`.venv`) so everyone develops against the same set of dependencies, listed in [`requirements.txt`](requirements.txt).

### Prerequisites
* Python 3.10+ installed and available on your PATH.

### Create the virtual environment

**Windows (PowerShell):**
```powershell
.\setup_venv.ps1
```
If script execution is blocked by your PowerShell policy, run:
```powershell
powershell -ExecutionPolicy Bypass -File .\setup_venv.ps1
```

**macOS / Linux (bash):**
```bash
./setup_venv.sh
```

Either script will:
1. Create a `.venv` folder in the repo root (if it doesn't already exist).
2. Upgrade `pip`.
3. Install every package listed in `requirements.txt`.

### Activate the environment

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

**macOS / Linux (bash):**
```bash
source .venv/bin/activate
```

### Adding new dependencies

When you need a new package, install it inside the activated venv, then add it to `requirements.txt` so the rest of the team picks it up next time they run the setup script.

### Visualización de resultados

Este proyecto incluye un script de visualización que genera gráficos modernos y profesionales a partir de la subred usada para Max-Cut:

```powershell
python -m src.visualize
```

El script produce las siguientes figuras en `figures/`:

* `red_nacional.png` — mapa de la red eléctrica nacional con la subred resaltada.
* `subred_valle_central.png` — visualización de la subred usada para los algoritmos.
* `comparacion_algoritmos.png` — barra comparativa de los valores de Max-Cut obtenidos por los métodos clásicos.
* `maxcut_partition.png` — partición del mejor algoritmo con los bordes cortados en rojo.

### Key dependencies

* [`pytket`](https://tket.quantinuum.com/) — quantum SDK / circuit compilation
* [`guppylang`](https://github.com/CQCL/guppylang) — quantum programming language (Guppy)
* [`numpy`](https://numpy.org/) — numerical arrays and scientific computing
* [`scipy`](https://scipy.org/) — scientific computing
* [`optax`](https://optax.readthedocs.io/) — gradient-based optimization
* [`cvxpy`](https://www.cvxpy.org/) — convex optimization
* [`networkx`](https://networkx.org/) — graph algorithms
* [`matplotlib`](https://matplotlib.org/) — plotting and visualization
* [`pytest`](https://pytest.org/) — testing framework

