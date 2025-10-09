# Noise-Aware QAOA (Max-Cut)

Compact benchmark suite for exploring noisy QAOA on random Erdős–Rényi Max-Cut instances. Compare warm-starts, mixers, optimizers, mitigation, and zero-noise extrapolation while capturing CSV/JSON artifacts, plots, and transpilation metrics.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Sanity check (≈30 s):

```bash
python -c "import noise_aware_qaoa as m; print('OK', len(m.run_suite_maxcut_fast(seed=7)['records']))"
```

Full `run_suite_maxcut()` with defaults (n=8, p=[1,2,3], 3 ablations) takes ~5–10 minutes. Use `run_suite_maxcut_fast()` when iterating.

## Minimal Python Usage

```python
from noise_aware_qaoa import (
    run_suite_maxcut_fast,
    run_suite_maxcut,
    export_artifacts,
    plot_approx_ratio,
)

# Fast mode (~30 s): reduced steps, n=6, single ablation
bundle = run_suite_maxcut_fast(seed=7)
df = export_artifacts(bundle, "qaoa_fast.csv", "qaoa_fast.json")
plot_approx_ratio(df, "qaoa_fast.png")

# Full benchmark (5–10 min): default n=8, p=[1,2,3], 3 ablations
full_bundle = run_suite_maxcut(seed=7)
print(len(full_bundle["records"]))
```

`bundle` dictionaries expose `records` (experiment rows), `layout` (transpilation stats), and `config` (global params).

## CLI

- Default sweep (n=8, p∈{1,2,3}):

  ```bash
  python noise_aware_qaoa.py
  ```

- Fast mode (reduced steps & ablations):

  ```bash
  python noise_aware_qaoa.py --fast --n 6 --p-list 1
  ```

- Custom example:

  ```bash
  python noise_aware_qaoa.py \
    --n 10 --p-list 1 2 3 4 \
    --graph-p 0.5 --shots 1024 \
    --p1 0.001 --p2 0.005 --readout 0.02 \
    --seed 42 \
    --csv qaoa_custom.csv --json qaoa_custom.json --fig qaoa_custom.png
  ```

Add `--help` to inspect every CLI flag.

## Output Artifacts

| File                  | Purpose                              |
| --------------------- | ------------------------------------ |
| `qaoa_records.csv`    | Flat table of all experiment records |
| `qaoa_records.json`   | Bundle with config + layout metrics  |
| `qaoa_ratio_vs_p.png` | Approximation ratio vs depth (`p`)   |

## Notebook

Open `example_usage.ipynb` for:

- Minimal p-sweeps with artifact export
- Fast vs full benchmark comparisons
- Mitigation & zero-noise extrapolation demos

Launch after installing requirements and attaching a Jupyter kernel in the same environment.

## Testing

Run the suite:

```bash
pytest -v test_noise_aware_qaoa.py
```

Handy variants:

- ```bash
  pytest -k "fast" -v
  ```
  Exercise the lightweight scenarios only.
- ```bash
  pytest test_noise_aware_qaoa.py::TestRunSuite::test_fast_bundle -v
  ```
  Focus on a single case while iterating.
- ```bash
  pytest -x -v
  ```
  Stop after the first failure.
- ```bash
  pytest -v --cov=noise_aware_qaoa --cov-report=term-missing
  ```
  Collect coverage when preparing releases.

## Noise Model (Defaults)

- 1-qubit depolarizing: `p1 = 1e-3`
- 2-qubit depolarizing: `p2 = 5 × p1`
- Readout flips: `readout_p = 0.02`

Override via `run_suite_maxcut` args or CLI flags `--p1`, `--p2`, `--readout`.

## ⚡ Performance Notes

- Fast mode (`run_suite_maxcut_fast` / `--fast`) trims optimizer steps to 10 and uses n=6, p=[1]; ideal for CI or quick validation.
- Standard mode (`run_suite_maxcut`) keeps 60 steps, three ablations, and higher shots; expect 5–10+ minutes on a laptop CPU.
- Runtime scales with qubits (`n`), depth list (`p`), ablations, and `shots`. Reduce any of these to experiment faster.

## Dependencies

See `requirements.txt` for pinned versions (Qiskit 2.2+, qiskit-aer, NumPy, Pandas, Matplotlib). Install hardware or GPU backends separately if needed.

## Contributing

PRs welcome—keep changes tested (unit tests or notebook snippets).

## License

MIT License (`LICENSE`).

## Citation

```bibtex
@software{noise_aware_qaoa,
  title={Noise-Aware QAOA Mini-Suite},
  year={2025},
  url={https://github.com/yourusername/noise-aware-qaoa}
}
```

---

If this toolkit helps your QAOA studies, a ⭐️ is always appreciated!
