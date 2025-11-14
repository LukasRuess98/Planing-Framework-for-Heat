from __future__ import annotations
from typing import Dict, Any, List, Optional
import os, json, time
import pandas as pd

try:
    import pyomo.environ as pyo
    HAVE_PYOMO = True
except Exception:
    HAVE_PYOMO = False
    pyo = None

from energis.config.merge import load_and_merge
from energis.io.loader import load_input_excel
from energis.models.system_builder import build_model

def excel_safe(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    if hasattr(df2.index, "tz") and df2.index.tz is not None:
        df2.index = df2.index.tz_convert("Europe/Berlin").tz_localize(None)
    return df2

def _estimate_max_thermal_capacity(cfg: dict) -> float:
    """Schätzt die maximale thermische Kapazität des Systems"""
    syscfg = cfg.get("system", {})
    cap = 0.0

    # Heat pumps (Summe max_th_mw)
    for hp in syscfg.get("heat_pumps", []):
        if hp.get("enabled", True):
            cap += float(hp.get("max_th_mw", 0.0))

    # Storage liefert nicht dauerhaft Leistung → kein thermischer Kapazitätsbeitrag

    # Generators (thermische caps) - FIXED: P2H wird hier nicht mehr doppelt gezählt
    gens = syscfg.get("generators", {})
    for key, par in gens.items():
        if par.get("enabled", False):
            cap += float(par.get("cap_th_mw", 0.0))

    return cap

def _assert_capacity_vs_demand(df: pd.DataFrame, cfg: dict, safety: float = 1.05):
    """Prüft ob genug thermische Kapazität für Demand-Peak vorhanden ist"""
    peak_demand = float(df["waermebedarf_MWth"].max())
    cap = _estimate_max_thermal_capacity(cfg)
    if cap < safety * peak_demand and peak_demand > 0:
        raise RuntimeError(
            f"Thermische Maximalleistung ({cap:.2f} MW_th) < {safety:.0%} des Demand-Peaks ({peak_demand:.2f} MW_th). "
            "Aktiviere/erhöhe Kapazitäten (HP/generators) oder reduziere Demand – sonst kann das Modell entweder nicht "
            "lösen oder produziert '0' bei Nullbedarf."
        )

def run_all(config_paths: List[str], overrides: Optional[Dict[str,Any]]=None) -> Dict[str,Any]:
    cfg = load_and_merge(config_paths)
    if overrides:
        cfg = {**cfg, **overrides}

    site = cfg.get("site", {})
    run  = cfg.get("run", {})
    dt_h = float(run.get("dt_h", 1.0))

    df = load_input_excel(site.get("input_xlsx","Import_Data.xlsx"), site, dt_hours=dt_h)

    # OPTIONAL: Kapazitätsprüfung aktivieren (auskommentiert für Flexibilität)
    # _assert_capacity_vs_demand(df, cfg, safety=1.05)
    
    m = build_model(df, cfg, dt_h=dt_h)
    if HAVE_PYOMO and m is not None:
        solver = run.get("solver", "glpk")
        try:
            opt = pyo.SolverFactory(solver)
        except Exception:
            opt = pyo.SolverFactory("glpk")
        opt.solve(m, tee=False)

        # Export timeseries minimal (grid + dump + any var with suffix patterns)
        ts = pd.DataFrame(index=df.index)
        ts["P_buy"]  = [pyo.value(m.P_buy[t]) for t in m.t]
        ts["P_sell"] = [pyo.value(m.P_sell[t]) for t in m.t]
        ts["Q_dump"] = [pyo.value(m.Q_dump[t]) for t in m.t]
        # capture component vars by naming convention
        for v in m.component_objects(pyo.Var, descend_into=True):
            name = v.name
            if name in ("P_buy","P_sell","Q_dump"): 
                continue
            try:
                ts[name] = [pyo.value(v[t]) for t in m.t]
            except Exception:
                pass

        costs = {"OBJ_value_EUR": float(pyo.value(m.obj)),
                 "P_buy_peak_MW": float(max(ts["P_buy"].max(), 0.0))}

    else:
        ts = df[["strompreis_EUR_MWh"]].copy()
        ts["P_buy"]=0.0; ts["P_sell"]=0.0; ts["Q_dump"]=0.0
        costs = {"OBJ_value_EUR": 0.0, "P_buy_peak_MW": 0.0}

    # Exports
    stamp = time.strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join("exports", f"{stamp}_PF_THEN_RH-Baseline")
    os.makedirs(outdir, exist_ok=True)
    scen_xlsx = os.path.join(outdir, "scenario.xlsx")
    with pd.ExcelWriter(scen_xlsx, engine="openpyxl") as xw:
        excel_safe(pd.DataFrame(list(cfg.get("meta", {"config_hash": ""}).items()), columns=["key","value"])).to_excel(xw, "meta", index=False)
        excel_safe(ts).to_excel(xw, "timeseries")
        pd.Series(costs).to_frame("value").to_excel(xw, "costs")
    # persist merged config dump
    with open(os.path.join(outdir,"merged_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    return {"scenario_xlsx": scen_xlsx, "outdir": outdir, "costs": costs}