Language and SDK. Python 3.10+ as the host, Guppy 0.21 for the quantum kernel, and Quantinuum's stack: pytket, local selene-sim, and qnexus in the cloud.



\*\*What worked.\*\* Guppy reads as typed Python, so the QAOA kernel lives alongside the rest of the pipeline (graph → qubo → qaoa) without leaving the language. Compilation to HUGR proved backend-agnostic: the same kernel runs on local Selene and on Nexus-hosted Selene, and both return a QsysResult, so count and energy decoding never changed. The Python ecosystem (SciPy/COBYLA, cvxpy, networkx) let us build the classical baselines and the benchmark in the same repo.



\*\*What didn't.\*\* Guppy 0.21's angle convention — angle(x) means half-turns, not radians — forced folding a 1/π factor at compile time. Guppy also cannot infer the type of an empty comptime list, so a zero coefficient had to be injected as a placeholder.



\*\*What was missing.\*\* A reference example with a weighted Hamiltonian and single-Z fields: the max-cut one is unweighted and maximizes, so the phase layer was generalized by hand. And on Nexus, batched submission: one job per objective evaluation, plus interactive login, kept that path out of the reproducible pipeline. Guppy also was lacking support for multi threading, which meant that for efficiency we ended up having to create individual OS processes to host several instances of guppy in Selene.

