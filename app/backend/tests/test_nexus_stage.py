import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import nexus_stage


def test_kernel_source_structure():
    h = [0.5, 0.0, -0.25]
    J = {(0, 1): 1.0, (1, 2): -2.0}
    src = nexus_stage.generate_kernel_source(h, J, gamma=0.3, beta=1.1, n=3)
    assert "@guppy" in src and "def main() -> None:" in src
    assert src.count("qubit()") == 3
    assert src.count("cx(") == 4              # 2 per ZZ term
    assert 'result("x0"' in src and 'result("x2"' in src
    # h[1] == 0 emits no single-Z rotation for qubit 1.
    assert "rz(q1," not in src.split("cx")[0]


def test_kernel_source_compiles_to_hugr():
    h = [0.1, -0.2]
    J = {(0, 1): 0.7}
    src = nexus_stage.generate_kernel_source(h, J, gamma=0.2, beta=0.9, n=2)
    kernel = nexus_stage.load_kernel(src)
    pkg = kernel.compile()
    assert pkg.modules
