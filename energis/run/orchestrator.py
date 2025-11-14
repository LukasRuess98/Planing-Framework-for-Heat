from __future__ import annotations

from typing import Dict, Any, List, Optional
import calendar
from datetime import datetime
import os
import json
import re
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


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _write_timeseries(path: str, table: TimeSeriesTable, extra: Dict[str, List[float]]) -> None:
    columns = ["timestamp"] + table.columns + list(extra.keys())
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(";".join(columns) + "\n")
        for idx, ts in enumerate(table.index):
            base = [ts.isoformat()]
            base.extend(str(table[col][idx]) for col in table.columns)
            base.extend(str(extra[name][idx]) for name in extra)
            handle.write(";".join(base) + "\n")


def _slugify(value: str | None) -> str:
    if not value:
        return "scenario"
    slug = re.sub(r"[^0-9a-zA-Z_-]+", "-", value.strip())
    slug = re.sub(r"-+", "-", slug).strip("-_")
    return slug or "scenario"


def _slice_table(table: TimeSeriesTable, indices: List[int]) -> TimeSeriesTable:
    new_index = [table.index[i] for i in indices]
    new_data = {col: [table[col][i] for i in indices] for col in table.columns}
    return TimeSeriesTable(new_index, table.columns[:], new_data)


def _parse_ts(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("Leerwert kann nicht als Datum interpretiert werden")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%d.%m.%Y %H:%M", "%d.%m.%Y"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Kann Datum/Uhrzeit nicht interpretieren: {value!r}")


def _apply_horizon(table: TimeSeriesTable, scenario_cfg: Dict[str, Any], dt_h: float) -> TimeSeriesTable:
    horizon = scenario_cfg.get("horizon")
    if not isinstance(horizon, dict):
        return table

    htype = str(horizon.get("type", "")).lower()
    if htype == "full_year":
        year = horizon.get("year")
        if year is None:
            if not table.index:
                raise RuntimeError("Zeitscheiben leer – kein Jahr auswählbar")
            year = table.index[0].year
        year = int(year)
        enforce = bool(horizon.get("enforce", True))
        indices = [i for i, ts in enumerate(table.index) if ts.year == year]
        if not indices:
            raise RuntimeError(f"Im Jahr {year} wurden keine Zeitschritte gefunden.")
        subset = _slice_table(table, indices)
        subset.ensure_frequency(dt_h)
        expected_hours = 8784 if calendar.isleap(year) else 8760
        actual_hours = len(subset) * dt_h
        diff = abs(actual_hours - expected_hours)
        if diff > 1e-6 and enforce:
            raise RuntimeError(
                f"Zeitraum deckt nicht das komplette Jahr {year} ab (erwartet {expected_hours} Stunden, gefunden {actual_hours})."
            )
        if diff > 1e-6 and not enforce:
            print(
                f"[SCENARIO] Hinweis: Daten decken nur {actual_hours} von {expected_hours} Stunden ab (enforce=false)."
            )
        print(f"[SCENARIO] Volles Jahr {year}: {len(subset)} Schritte ({subset.index[0]} → {subset.index[-1]})")
        return subset

    start = horizon.get("start")
    end = horizon.get("end")
    if start is None and end is None:
        return table

    start_dt = _parse_ts(start) if start is not None else None
    end_dt = _parse_ts(end) if end is not None else None
    if start_dt and end_dt and start_dt > end_dt:
        raise RuntimeError("Ungültiger Zeithorizont: Start liegt nach dem Ende")
    indices = []
    for i, ts in enumerate(table.index):
        if start_dt and ts < start_dt:
            continue
        if end_dt and ts > end_dt:
            continue
        indices.append(i)
    if not indices:
        raise RuntimeError("Zeithorizont enthält keine Zeitschritte")
    subset = _slice_table(table, indices)
    subset.ensure_frequency(dt_h)
    print(f"[SCENARIO] Zeitraum {subset.index[0]} → {subset.index[-1]} ({len(subset)} Schritte)")
    return subset


def run_all(config_paths: List[str], overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = load_and_merge(config_paths)
    if overrides:
        cfg = _deep_update(cfg, overrides)

    site = cfg.get("site", {})
    run_cfg = cfg.get("run", {})
    dt_h = float(run_cfg.get("dt_h", 1.0))

    table = load_input_excel(site.get("input_xlsx", "Import_Data.xlsx"), site, dt_hours=dt_h)
    table.ensure_frequency(dt_h)

    scenario_cfg = cfg.get("scenario", {})
    table = _apply_horizon(table, scenario_cfg, dt_h)
    _assert_capacity_vs_demand(table, cfg)

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
    mode = str(scenario_cfg.get("mode", "PF"))
    title = str(scenario_cfg.get("title", "Baseline"))
    tag = scenario_cfg.get("tag") or f"{mode}-{title}"
    outdir = os.path.join("exports", f"{stamp}_{_slugify(tag)}")
    os.makedirs(outdir, exist_ok=True)
    scen_file = os.path.join(outdir, "scenario_timeseries.csv")
    _write_timeseries(scen_file, table, ts)

    with open(os.path.join(outdir, "costs.json"), "w", encoding="utf-8") as handle:
        json.dump(costs, handle, indent=2)

    with open(os.path.join(outdir, "merged_config.json"), "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2)

    return {"scenario_xlsx": scen_file, "outdir": outdir, "costs": costs, "scenario": scenario_cfg}

