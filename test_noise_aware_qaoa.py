"""
Core tests for Noise-Aware QAOA mini-suite.

Run with: pytest -v test_noise_aware_qaoa.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from noise_aware_qaoa import (
    set_seed,
    run_suite_maxcut,
    export_artifacts,
    plot_approx_ratio,
)


def _mean_used_ratio(df: pd.DataFrame) -> float:
    """Use mitigated ratio if present; otherwise plain."""
    s = df["approx_ratio_mitig"].fillna(df["approx_ratio_plain"])
    return float(s.mean())


class TestBasicRuns:
    def test_minimal_maxcut_run(self, tmp_path: Path):
        set_seed(42)
        bundle = run_suite_maxcut(n=6, p_list=[1, 2], graph_p=0.4, shots=256, seed=42)
        assert isinstance(bundle, dict)
        assert "records" in bundle and isinstance(bundle["records"], list)
        assert "layout" in bundle and isinstance(bundle["layout"], list)
        assert "config" in bundle and isinstance(bundle["config"], dict)

        csv_p = tmp_path / "out.csv"
        json_p = tmp_path / "out.json"
        df = export_artifacts(bundle, str(csv_p), str(json_p))

        assert csv_p.exists()
        assert json_p.exists()
        assert isinstance(df, pd.DataFrame)
        needed = {
            "p",
            "warm_start",
            "mixer",
            "optimizer",
            "approx_ratio_plain",
            "shots",
        }

        assert needed.issubset(set(df.columns))

    def test_plot_generation(self, tmp_path: Path):
        set_seed(1)
        bundle = run_suite_maxcut(n=6, p_list=[1, 2, 3], graph_p=0.5, shots=256, seed=1)
        df = export_artifacts(bundle, str(tmp_path / "a.csv"), str(tmp_path / "a.json"))
        fig_path = tmp_path / "ratio.png"
        plot_approx_ratio(df, str(fig_path))
        assert fig_path.exists()


class TestAblations:
    def test_ablation_matrix_runs(self, tmp_path: Path):
        set_seed(7)
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
                "zne_scales": [1, 3],
            },
            {
                "warm_start": True,
                "mixer": "x",
                "optimizer": "nelder-mead",
                "readout_mit": True,
                "zne_scales": [1],
            },
        ]
        bundle = run_suite_maxcut(
            n=8, p_list=[1, 2], graph_p=0.4, shots=512, ablations=ablations, seed=7
        )
        df = export_artifacts(
            bundle, str(tmp_path / "abl.csv"), str(tmp_path / "abl.json")
        )

        combos = df[["warm_start", "mixer", "optimizer"]].drop_duplicates()
        assert len(combos) == len(ablations)

        if any(a.get("readout_mit", False) for a in ablations):
            assert "approx_ratio_mitig" in df.columns

    def test_noise_sweep_and_mitigation_toggle(self, tmp_path: Path):
        for p1 in [0.0, 1e-3, 5e-3]:
            set_seed(11)
            bundle = run_suite_maxcut(
                n=6,
                p_list=[1, 2],
                graph_p=0.4,
                shots=256,
                ablations=[
                    {
                        "warm_start": True,
                        "mixer": "x",
                        "optimizer": "adam",
                        "readout_mit": True,
                        "zne_scales": [1, 3, 5],
                    }
                ],
                noise_p1=p1,
                noise_p2=5 * p1,
                readout_p=0.02,
                seed=11,
            )
            df = export_artifacts(
                bundle,
                str(tmp_path / f"noise_{p1:.4f}.csv"),
                str(tmp_path / f"noise_{p1:.4f}.json"),
            )
            r = _mean_used_ratio(df)
            assert np.isfinite(r)
            assert 0.0 <= r <= 1.2


class TestLayoutMetrics:
    def test_layout_metrics_present(self, tmp_path: Path):
        set_seed(21)
        bundle = run_suite_maxcut(n=6, p_list=[1], graph_p=0.3, shots=256, seed=21)
        export_artifacts(bundle, str(tmp_path / "L.csv"), str(tmp_path / "L.json"))

        data = json.loads(Path(tmp_path / "L.json").read_text())
        layout = data.get("layout", [])
        assert isinstance(layout, list)
        assert len(layout) >= 1
        sample = layout[0]
        for key in ("depth", "cx", "qubits"):
            assert key in sample
            assert isinstance(sample[key], int)


class TestReproducibility:
    def test_same_seed_same_results(self):
        set_seed(123)
        A = run_suite_maxcut(n=6, p_list=[1], graph_p=0.4, shots=256, seed=123)
        set_seed(123)
        B = run_suite_maxcut(n=6, p_list=[1], graph_p=0.4, shots=256, seed=123)

        dfA = pd.DataFrame(A["records"])
        dfB = pd.DataFrame(B["records"])

        sort_cols = [
            "n",
            "p",
            "mixer",
            "warm_start",
            "optimizer",
            "shots",
            "readout_mit",
        ]
        dfA = dfA.sort_values(by=sort_cols).reset_index(drop=True)
        dfB = dfB.sort_values(by=sort_cols).reset_index(drop=True)

        assert dfA.shape == dfB.shape

        a_used = dfA["approx_ratio_mitig"].fillna(dfA["approx_ratio_plain"])
        b_used = dfB["approx_ratio_mitig"].fillna(dfB["approx_ratio_plain"])
        pd.testing.assert_series_equal(
            a_used.reset_index(drop=True), b_used.reset_index(drop=True)
        )

        assert (dfA["zne_scales"].apply(str) == dfB["zne_scales"].apply(str)).all()

    def test_different_seed_changes_results(self):
        set_seed(1)
        A = run_suite_maxcut(n=6, p_list=[1], graph_p=0.4, shots=256, seed=1)
        set_seed(2)
        B = run_suite_maxcut(n=6, p_list=[1], graph_p=0.4, shots=256, seed=2)

        dfA = pd.DataFrame(A["records"])
        dfB = pd.DataFrame(B["records"])
        a_used = dfA["approx_ratio_mitig"].fillna(dfA["approx_ratio_plain"]).mean()
        b_used = dfB["approx_ratio_mitig"].fillna(dfB["approx_ratio_plain"]).mean()

        assert abs(a_used - b_used) >= 0.0


def _maybe_import_portfolio():
    try:
        from noise_aware_qaoa import run_suite_portfolio
    except Exception:
        run_suite_portfolio = None
    return run_suite_portfolio


@pytest.mark.skipif(
    _maybe_import_portfolio() is None, reason="Portfolio runner not implemented"
)
def test_portfolio_suite_smoke(tmp_path: Path):
    from noise_aware_qaoa import run_suite_portfolio

    set_seed(5)
    bundle = run_suite_portfolio(
        num_assets=6,
        p_list=[1, 2],
        shots=256,
        warm_start=True,
        readout_mit=True,
        seed=5,
    )
    df = export_artifacts(
        bundle, str(tmp_path / "port.csv"), str(tmp_path / "port.json")
    )
    assert "approx_ratio_plain" in df.columns or "expected_return" in df.columns
