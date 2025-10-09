"""Noise-Aware QAOA Mini-Suite: benchmarks QAOA for Max-Cut with various
ablations and mitigation strategies.

Features: QAOA for Max-Cut on random Erdős–Rényi graphs, warm-start strategies,
mixer choices, multiple optimizers (Adam, Nelder-Mead), readout error mitigation,
zero-noise extrapolation (ZNE), layout-aware transpilation statistics, data export
(CSV/JSON) and visualization. Tested on Qiskit 2.2.1+.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import argparse
import json
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


_OPTIMIZE_STEPS = 60


def set_seed(seed: int = 7):
    """Set random seed for reproducibility."""
    np.random.seed(int(seed))
    print(f"[repro] numpy.random.seed = {seed}")


def simple_noise_model(
    p1: float = 1e-3, p2: float = 5e-3, readout_p: float = 0.02
) -> NoiseModel:
    """Build simple depolarizing + readout noise model."""
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["rz", "sx", "x", "id"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    ro = ReadoutError([[1 - readout_p, readout_p], [readout_p, 1 - readout_p]])
    nm.add_all_qubit_readout_error(ro)
    return nm


def counts_to_expval_zpauli(counts: Dict[str, int], pauli: str) -> float:
    """Calculate expectation value of Z-string Pauli from bitstring counts."""
    n = len(next(iter(counts)))
    assert len(pauli) == n
    tot = sum(counts.values())
    exp = 0.0
    for bitstr, c in counts.items():
        val = 1.0
        for i, ch in enumerate(pauli):
            if ch == "I":
                continue
            z = 1 if bitstr[n - 1 - i] == "0" else -1
            val *= z
        exp += val * c
    return exp / tot if tot else 0.0


def random_maxcut_hamiltonian(
    n: int, p_edge: float = 0.4, w_low: float = 1.0, w_high: float = 1.0
) -> Tuple[SparsePauliOp, List[Tuple[int, int]], np.ndarray]:
    """Generate random Max-Cut problem on Erdős–Rényi graph."""
    edges, weights = [], []
    zterms = {}
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < p_edge:
                w = float(np.random.uniform(w_low, w_high))
                edges.append((i, j))
                weights.append(w)
                label = ["I"] * n
                label[i] = "Z"
                label[j] = "Z"
                key = "".join(label)
                zterms[key] = zterms.get(key, 0.0) + (-w)

    if not zterms:
        i, j = 0, min(1, n - 1)
        edges = [(i, j)]
        weights = [1.0]
        key = "".join(["Z" if k in (i, j) else "I" for k in range(n)])
        zterms[key] = -1.0

    labels = list(zterms.keys())
    coeffs = np.array([zterms[k] for k in labels], dtype=float)
    return (
        SparsePauliOp.from_list([(lab, c) for lab, c in zip(labels, coeffs)]),
        edges,
        np.array(weights),
    )


def qaoa_ansatz(
    H: SparsePauliOp, p: int, mixer: str = "x", warm_start: np.ndarray | None = None
) -> Tuple[QuantumCircuit, List[Parameter], List[Parameter]]:
    """Build QAOA ansatz circuit."""
    n = H.num_qubits
    gammas = ParameterVector("γ", p)
    betas = ParameterVector("β", p)

    qc = QuantumCircuit(n, n, name=f"QAOA(p={p},{mixer})")

    if warm_start is None:
        qc.h(range(n))
    else:
        for i in range(n):
            theta = np.pi * float(warm_start[i])
            qc.ry(theta, i)

    for layer in range(p):
        for label, coeff in zip(H.paulis.to_labels(), H.coeffs):
            theta = 2 * gammas[layer] * float(coeff.real)
            idx = [q for q, s in enumerate(label) if s == "Z"]

            if len(idx) == 1:
                qc.rz(theta, idx[0])
            elif len(idx) >= 2:
                for a, b in zip(idx[:-1], idx[1:]):
                    qc.cx(a, b)
                qc.rz(theta, idx[-1])
                for a, b in list(zip(idx[:-1], idx[1:]))[::-1]:
                    qc.cx(a, b)

        if mixer == "x":
            for q in range(n):
                qc.rx(2 * betas[layer], q)
        else:
            for q in range(n):
                qc.rx(2 * betas[layer], q)

    qc.measure(range(n), range(n))

    return qc, list(gammas), list(betas)


def calibrate_readout_matrix(sim: AerSimulator, n: int) -> np.ndarray:
    """Calibrate readout error matrix (simplified tensor-product model)."""
    ro = np.array([[0.98, 0.02], [0.02, 0.98]])

    M = np.array([[1.0]])
    for _ in range(n):
        M = np.kron(M, ro)
    return M


def mitigate_counts_readout(
    counts: Dict[str, int], M_inv: np.ndarray
) -> Dict[str, float]:
    """Apply inverse readout matrix to mitigate measurement errors."""
    keys = sorted(counts.keys())
    n = len(keys[0])
    vec = np.zeros(2**n)

    for b, c in counts.items():
        idx = int(b, 2)
        vec[idx] = c
    vec = vec / max(vec.sum(), 1.0)

    mitig = M_inv @ vec
    mitig = np.clip(mitig, 0, 1)
    mitig = mitig / max(mitig.sum(), 1e-12)

    out = {}
    for i, p in enumerate(mitig):
        out[format(i, f"0{n}b")] = float(p)
    return out


def richardson_extrapolate(scales: List[int], vals: List[float]) -> float:
    """Zero-noise extrapolation using Richardson extrapolation."""
    x = np.array(scales, dtype=float)
    y = np.array(vals, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = sol
    return float(intercept)


@dataclass
class QAOAConfig:
    """Configuration for QAOA run."""

    n: int
    p: int
    mixer: str
    warm_start: bool
    optimizer: str
    shots: int
    readout_mit: bool
    zne_scales: List[int]
    problem: str


def _run_single(sim: AerSimulator, qc: QuantumCircuit, shots: int) -> Dict[str, int]:
    """Execute circuit on simulator."""
    tqc = transpile(qc, backend=sim, optimization_level=1)
    res = sim.run(tqc, shots=shots).result()
    return res.get_counts()


def _approx_ratio_maxcut(
    counts: Dict[str, int], edges: List[Tuple[int, int]], weights: np.ndarray
) -> float:
    """Calculate Max-Cut approximation ratio from measurement counts."""
    if not counts:
        return 0.0
    tot = sum(counts.values())

    def cut_value(bitstr):
        val = 0.0
        for k, (i, j) in enumerate(edges):
            val += weights[k] * (1 if bitstr[i] != bitstr[j] else 0)
        return val

    exp = sum(cut_value(b) * c for b, c in counts.items()) / tot
    opt_ub = float(weights.sum())
    return float(exp / max(opt_ub, 1e-12))


def _warm_start_vector_maxcut(n: int) -> np.ndarray:
    """Generate random warm-start vector."""
    return np.random.rand(n)


def optimize_angles(
    p: int,
    cost_fun,
    method: str = "adam",
    steps: int | None = None,
    lr: float = 0.2,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Optimize QAOA parameters."""
    if steps is None:
        steps = _OPTIMIZE_STEPS

    rng = np.random.default_rng(seed)
    gam = rng.normal(scale=0.1, size=p)
    bet = rng.normal(scale=0.1, size=p)

    if method.lower() == "nelder-mead":
        x = np.concatenate([gam, bet])
        best = cost_fun(x)
        xb = x.copy()
        max_iters = min(80, steps * 2)
        for _ in range(max_iters):
            dx = rng.normal(scale=0.15, size=2 * p)
            x2 = x + dx
            v2 = cost_fun(x2)
            if v2 < best:
                best, xb = v2, x2
                x = x2
            else:
                x = x - 0.5 * dx
        g, b = xb[:p], xb[p:]
        return g, b, best

    x = np.concatenate([gam, bet])
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    eps = 1e-8
    beta1, beta2 = 0.9, 0.999

    def grad(f, z, h=1e-3):
        g = np.zeros_like(z)
        for i in range(len(z)):
            z[i] += h
            f1 = f(z)
            z[i] -= 2 * h
            f2 = f(z)
            z[i] += h
            g[i] = (f1 - f2) / (2 * h)
        return g

    val = cost_fun(x)
    for t in range(1, steps + 1):
        g = grad(cost_fun, x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        mhat = m / (1 - beta1**t)
        vhat = v / (1 - beta2**t)
        x = x - lr * mhat / (np.sqrt(vhat) + eps)
        val = cost_fun(x)

    g, b = x[:p], x[p:]
    return g, b, val


def run_suite_maxcut(
    n: int = 8,
    p_list: List[int] = [1, 2, 3],
    graph_p: float = 0.4,
    ablations: List[Dict[str, Any]] | None = None,
    shots: int = 1024,
    noise_p1: float = 1e-3,
    noise_p2: float = 5e-3,
    readout_p: float = 0.02,
    seed: int = 7,
    coupling_map: List[Tuple[int, int]] | None = None,
) -> Dict[str, Any]:
    """Run comprehensive QAOA benchmark suite for Max-Cut."""
    set_seed(seed)
    H, edges, weights = random_maxcut_hamiltonian(n, p_edge=graph_p)
    nm = simple_noise_model(noise_p1, noise_p2, readout_p)
    sim = AerSimulator(noise_model=nm, seed_simulator=seed)

    if ablations is None:
        ablations = [
            {
                "warm_start": False,
                "mixer": "x",
                "optimizer": "adam",
                "readout_mit": False,
                "zne_scales": [1],
            },
            {
                "warm_start": True,
                "mixer": "x",
                "optimizer": "adam",
                "readout_mit": True,
                "zne_scales": [1, 3, 5],
            },
            {
                "warm_start": True,
                "mixer": "x",
                "optimizer": "nelder-mead",
                "readout_mit": True,
                "zne_scales": [1],
            },
        ]

    records, layout_rows = [], []
    nqubits = H.num_qubits

    if any(ab.get("readout_mit") for ab in ablations):
        M = calibrate_readout_matrix(sim, nqubits)
        M_inv = np.linalg.pinv(M)

    for p in p_list:
        for ab in ablations:
            cfg = QAOAConfig(
                n=n,
                p=p,
                mixer=ab["mixer"],
                warm_start=ab["warm_start"],
                optimizer=ab["optimizer"],
                shots=shots,
                readout_mit=ab["readout_mit"],
                zne_scales=ab["zne_scales"],
                problem="maxcut",
            )

            warm = _warm_start_vector_maxcut(n) if cfg.warm_start else None
            qc, gam, bet = qaoa_ansatz(H, p=p, mixer=cfg.mixer, warm_start=warm)

            def objective(x):
                assign = {gam[i]: x[i] for i in range(p)}
                assign.update({bet[i]: x[p + i] for i in range(p)})
                qc_bound = qc.assign_parameters(assign)
                counts = _run_single(sim, qc_bound, shots=shots)
                ar = _approx_ratio_maxcut(counts, edges, weights)
                return -ar

            g_opt, b_opt, best = optimize_angles(
                p, objective, method=cfg.optimizer, seed=seed
            )
            assign = {gam[i]: g_opt[i] for i in range(p)}
            assign.update({bet[i]: b_opt[i] for i in range(p)})
            qc_best = qc.assign_parameters(assign)

            tr_backend = AerSimulator()
            tqc = transpile(
                qc_best,
                backend=tr_backend,
                optimization_level=1,
                coupling_map=coupling_map,
            )
            layout_rows.append(
                {
                    "p": p,
                    "mixer": cfg.mixer,
                    "warm_start": cfg.warm_start,
                    "depth": tqc.depth(),
                    "cx": int(tqc.count_ops().get("cx", 0)),
                    "qubits": tqc.num_qubits,
                }
            )

            counts = _run_single(sim, qc_best, shots=shots)
            approx_plain = _approx_ratio_maxcut(counts, edges, weights)

            approx_mitig = None
            if cfg.readout_mit:
                dist = mitigate_counts_readout(counts, M_inv)
                counts_m = {k: int(round(v * shots)) for k, v in dist.items()}
                approx_mitig = _approx_ratio_maxcut(counts_m, edges, weights)

            approx_zne = None
            if cfg.zne_scales and len(cfg.zne_scales) > 1:
                vals = []
                for s in cfg.zne_scales:
                    nm_s = simple_noise_model(noise_p1 * s, noise_p2 * s, readout_p)
                    sim_s = AerSimulator(noise_model=nm_s, seed_simulator=seed)
                    counts_s = _run_single(sim_s, qc_best, shots=shots)
                    vals.append(_approx_ratio_maxcut(counts_s, edges, weights))
                approx_zne = richardson_extrapolate(cfg.zne_scales, vals)

            records.append(
                {
                    **asdict(cfg),
                    "theta_dim": p,
                    "approx_ratio_plain": approx_plain,
                    "approx_ratio_mitig": approx_mitig,
                    "approx_ratio_zne": approx_zne,
                    "opt_obj": float(best),
                }
            )

    return {
        "records": records,
        "layout": layout_rows,
        "config": {
            "n": n,
            "p_list": p_list,
            "graph_p": graph_p,
            "shots": shots,
            "noise_p1": noise_p1,
            "noise_p2": noise_p2,
            "readout_p": readout_p,
            "seed": seed,
        },
    }


def run_suite_maxcut_fast(
    n: int = 6,
    p_list: List[int] = [1],
    ablations: List[Dict[str, Any]] | None = None,
    **kwargs,
) -> Dict[str, Any]:
    """Fast version of run_suite_maxcut for smoke testing."""
    global _OPTIMIZE_STEPS
    old_steps = _OPTIMIZE_STEPS
    _OPTIMIZE_STEPS = 10

    if ablations is None:
        ablations = [
            {
                "warm_start": False,
                "mixer": "x",
                "optimizer": "adam",
                "readout_mit": False,
                "zne_scales": [1],
            }
        ]

    kwargs.setdefault("shots", 256)
    kwargs.setdefault("graph_p", 0.4)

    try:
        result = run_suite_maxcut(n=n, p_list=p_list, ablations=ablations, **kwargs)
    finally:
        _OPTIMIZE_STEPS = old_steps

    return result


def export_artifacts(
    bundle: Dict[str, Any], csv_path: str, json_path: str
) -> pd.DataFrame:
    """Export benchmark results to CSV and JSON."""
    df = pd.DataFrame(bundle["records"])
    df.to_csv(csv_path, index=False)

    json_safe = {
        "config": bundle["config"],
        "layout": bundle["layout"],
        "records": bundle["records"],
    }

    with open(json_path, "w") as f:
        json.dump(json_safe, f, indent=2, default=float)

    print(f"✅ Saved CSV → {csv_path}")
    print(f"✅ Saved JSON → {json_path}")
    return df


def plot_approx_ratio(
    df: pd.DataFrame,
    out_path: str | None = None,
    title: str = "Max-Cut: Approx. Ratio vs p",
):
    """Plot approximation ratio vs QAOA depth."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for key, g in df.groupby(["warm_start", "mixer", "optimizer", "readout_mit"]):
        label = f"warm={key[0]}, opt={key[2]}, mit={key[3]}"
        y = g["approx_ratio_mitig"].fillna(g["approx_ratio_plain"])
        ax.plot(g["p"], y, marker="o", linestyle="-", label=label)

    ax.set_ylabel("Approximation Ratio")
    ax.set_xlabel("QAOA depth p")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"📈 Figure saved → {out_path}")

    plt.close()


def parse_args():
    """Parse command-line arguments."""
    pa = argparse.ArgumentParser(description="Noise-Aware QAOA Mini-Suite (Max-Cut)")
    pa.add_argument("--n", type=int, default=8, help="Number of qubits")
    pa.add_argument(
        "--p-list", type=int, nargs="+", default=[1, 2, 3], help="QAOA depths"
    )
    pa.add_argument("--graph-p", type=float, default=0.4, help="Edge probability")
    pa.add_argument("--shots", type=int, default=1024, help="Measurement shots")
    pa.add_argument("--p1", type=float, default=1e-3, help="1Q error rate")
    pa.add_argument("--p2", type=float, default=5e-3, help="2Q error rate")
    pa.add_argument("--readout", type=float, default=0.02, help="Readout error rate")
    pa.add_argument("--seed", type=int, default=7, help="Random seed")
    pa.add_argument("--csv", type=str, default="qaoa_records.csv", help="Output CSV")
    pa.add_argument("--json", type=str, default="qaoa_records.json", help="Output JSON")
    pa.add_argument(
        "--fig", type=str, default="qaoa_ratio_vs_p.png", help="Output figure"
    )
    pa.add_argument("--fast", action="store_true", help="Use fast mode for testing")
    return pa.parse_args()


def main():
    """Main execution function."""
    args = parse_args()

    print("=" * 70)
    print("NOISE-AWARE QAOA BENCHMARK SUITE")
    print("=" * 70)

    if args.fast:
        print("🚀 Running in FAST mode (reduced optimization steps)")
        bundle = run_suite_maxcut_fast(
            n=args.n,
            p_list=args.p_list,
            graph_p=args.graph_p,
            shots=args.shots,
            noise_p1=args.p1,
            noise_p2=args.p2,
            readout_p=args.readout,
            seed=args.seed,
        )
    else:
        bundle = run_suite_maxcut(
            n=args.n,
            p_list=args.p_list,
            graph_p=args.graph_p,
            shots=args.shots,
            noise_p1=args.p1,
            noise_p2=args.p2,
            readout_p=args.readout,
            seed=args.seed,
        )

    df = export_artifacts(bundle, args.csv, args.json)
    plot_approx_ratio(df, args.fig)

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"✨ Tested {len(df)} configurations")
    print(f"✨ Mean approximation ratio: {df['approx_ratio_plain'].mean():.3f}")


if __name__ == "__main__":
    main()
