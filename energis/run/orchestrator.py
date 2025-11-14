from __future__ import annotations

from typing import Dict, Any, List, Optional
import os
import json
import time

try:  # optional dependency
    import pyomo.environ as pyo
    HAVE_PYOMO = True
except Exception:  # pragma: no cover
    HAVE_PYOMO = False
    pyo = None

from energis.config.merge import load_and_merge
from energis.io.loader import load_input_excel
from energis.models.system_builder import build_model
from energis.utils.timeseries import TimeSeriesTable


def _estimate_max_thermal_capacity(cfg: dict) -> float:
    syscfg = cfg.get("system", {})
    cap = 0.0
    for hp in syscfg.get("heat_pumps", []):
        if hp.get("enabled", True):
            cap += float(hp.get("max_th_mw", 0.0))
    gens = syscfg.get("generators", {})
    for _, par in gens.items():
        if par.get("enabled", False):
            cap += float(par.get("cap_th_mw", 0.0))
    return cap


def _assert_capacity_vs_demand(table: TimeSeriesTable, cfg: dict, safety: float = 1.05) -> None:
    peak_demand = max(table["waermebedarf_MWth"])
    cap = _estimate_max_thermal_capacity(cfg)
    if cap < safety * peak_demand and peak_demand > 0:
        raise RuntimeError(
            "Thermische Maximalleistung zu gering für den Demand-Peak. Bitte Kapazitäten erhöhen."
        )


def _write_timeseries(path: str, table: TimeSeriesTable, extra: Dict[str, List[float]]) -> None:
    columns = ["timestamp"] + table.columns + list(extra.keys())
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(";".join(columns) + "\n")
        for idx, ts in enumerate(table.index):
            base = [ts.isoformat()]
            base.extend(str(table[col][idx]) for col in table.columns)
            base.extend(str(extra[name][idx]) for name in extra)
            handle.write(";".join(base) + "\n")


def run_all(config_paths: List[str], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = load_and_merge(config_paths)
    if overrides:
        cfg = {**cfg, **overrides}

    site = cfg.get("site", {})
    run_cfg = cfg.get("run", {})
    dt_h = float(run_cfg.get("dt_h", 1.0))

    table = load_input_excel(site.get("input_xlsx", "Import_Data.xlsx"), site, dt_hours=dt_h)

    m = build_model(table, cfg, dt_h=dt_h)
    if HAVE_PYOMO and m is not None:
        solver = run_cfg.get("solver", "glpk")
        try:
            opt = pyo.SolverFactory(solver)
        except Exception:
            opt = pyo.SolverFactory("glpk")
        opt.solve(m, tee=False)
        ts = {
            "P_buy": [pyo.value(m.P_buy[t]) for t in m.t],
            "P_sell": [pyo.value(m.P_sell[t]) for t in m.t],
            "Q_dump": [pyo.value(m.Q_dump[t]) for t in m.t],
        }
        costs = {
            "OBJ_value_EUR": float(pyo.value(m.obj)),
            "P_buy_peak_MW": float(max(ts["P_buy"], default=0.0)),
        }
    else:
        ts = {"P_buy": [0.0] * len(table), "P_sell": [0.0] * len(table), "Q_dump": [0.0] * len(table)}
        costs = {"OBJ_value_EUR": 0.0, "P_buy_peak_MW": 0.0}

    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("exports", f"{stamp}_PF_THEN_RH-Baseline")
    os.makedirs(outdir, exist_ok=True)
    scen_file = os.path.join(outdir, "scenario_timeseries.csv")
    _write_timeseries(scen_file, table, ts)

    with open(os.path.join(outdir, "costs.json"), "w", encoding="utf-8") as handle:
        json.dump(costs, handle, indent=2)

    with open(os.path.join(outdir, "merged_config.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)

    return {"scenario_xlsx": scen_file, "outdir": outdir, "costs": costs}

