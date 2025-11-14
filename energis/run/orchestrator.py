from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Any, List, Optional, Mapping, Sequence
import calendar
from datetime import datetime
import os
import json
import re
import time
import logging
import math

try:  # optional dependency
    import pyomo.environ as pyo
    HAVE_PYOMO = True
except Exception:  # pragma: no cover
    HAVE_PYOMO = False
    pyo = None

from energis.config.merge import load_and_merge
from energis.io.loader import load_input_excel
from energis.io.exporter import write_timeseries_csv, write_excel_workbook
from energis.models.system_builder import build_model
from energis.utils.timeseries import TimeSeriesTable


logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Return a JSON-serialisable representation of ``value``.

    The solver meta data pulled from Pyomo occasionally contains sentinel
    objects such as ``UndefinedData`` which the standard :mod:`json` module
    cannot serialise.  We normalise those values (and other non-primitive
    objects) so that exporting results to JSON never fails.
    """

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(val) for val in value]
    if HAVE_PYOMO and value.__class__.__name__ == "UndefinedData":  # pragma: no cover - depends on Pyomo
        return None
    return str(value)


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


def _gather_component_metadata(cfg: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "heat_pumps": [],
        "storage": None,
        "generators": [],
        "p2h": None,
    }

    syscfg = cfg.get("system", {})

    for hp in syscfg.get("heat_pumps", []):
        if not hp.get("enabled", True):
            continue
        meta["heat_pumps"].append(
            {
                "id": str(hp.get("id", "HP")),
                "max_th": float(hp.get("max_th_mw", 0.0)),
            }
        )

    sto_cfg = syscfg.get("storage", {})
    if sto_cfg.get("enabled", False):
        meta["storage"] = {
            "name": sto_cfg.get("id", "TES") or "TES",
            "e_max": float(sto_cfg.get("max_energy_mwh", 0.0)),
            "p_max": float(sto_cfg.get("max_power_mw", 0.0)),
        }

    fuels = cfg.get("fuels", {})
    gen_cfg = cfg.get("generators", {})

    for key, par in syscfg.get("generators", {}).items():
        if not par.get("enabled", False):
            continue
        if key == "p2h":
            meta["p2h"] = {
                "name": "P2H",
                "cap_th": float(par.get("cap_th_mw", 0.0)),
                "eff": float(gen_cfg.get("p2h", {}).get("el_to_th_eff", 0.0)),
            }
            continue

        gpar = gen_cfg.get(key, {})
        fuel_bus = gpar.get("fuel_bus", "gas")
        fuel_info = fuels.get(fuel_bus, {})
        meta["generators"].append(
            {
                "key": key,
                "name": key.upper(),
                "cap_th": float(par.get("cap_th_mw", 0.0)),
                "fuel_bus": fuel_bus,
                "fuel_price": float(fuel_info.get("price_eur_mwh", 0.0)),
                "fuel_emission": float(fuel_info.get("ef_kg_per_mwh_fuel", 0.0)),
                "has_el": gpar.get("el_eff") is not None,
            }
        )

    return meta


def _flatten_summary(sections: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for section, metrics in sections.items():
        for key, value in metrics.items():
            flat[f"{section}.{key}"] = value
    return flat


def _collect_timeseries_and_summary(
    table: TimeSeriesTable,
    cfg: Dict[str, Any],
    dt_h: float,
    model: Any | None,
) -> tuple[OrderedDict[str, List[float]], OrderedDict[str, OrderedDict[str, Any]], Dict[str, Any]]:
    meta = _gather_component_metadata(cfg)
    n = len(table)

    series: OrderedDict[str, List[float]] = OrderedDict()
    series["P_buy_MW"] = [0.0] * n
    series["P_sell_MW"] = [0.0] * n
    series["Q_dump_MWth"] = [0.0] * n

    for hp in meta["heat_pumps"]:
        comp = hp["id"]
        series[f"{comp}_Q_th_MW"] = [0.0] * n
        series[f"{comp}_Pel_MW"] = [0.0] * n
        series[f"{comp}_on"] = [0.0] * n

    if meta["storage"]:
        series["TES_SOC_MWh"] = [0.0] * n
        series["TES_charge_MW"] = [0.0] * n
        series["TES_discharge_MW"] = [0.0] * n

    for gen in meta["generators"]:
        comp = gen["name"]
        series[f"{comp}_Q_th_MW"] = [0.0] * n
        series[f"{comp}_fuel_MW"] = [0.0] * n
        if gen["has_el"]:
            series[f"{comp}_Pel_MW"] = [0.0] * n

    if meta["p2h"]:
        series["P2H_Q_th_MW"] = [0.0] * n
        series["P2H_Pel_MW"] = [0.0] * n

    objective = OrderedDict(
        [
            ("OBJ_value_EUR", 0.0),
            ("P_buy_peak_MW", 0.0),
            ("Grid_energy_cost_EUR", 0.0),
            ("Grid_sell_revenue_EUR", 0.0),
            ("Grid_net_cost_EUR", 0.0),
            ("Fuel_cost_EUR", 0.0),
            ("Fuel_emissions_t", 0.0),
            ("Dump_cost_EUR", 0.0),
            ("CO2_cost_EUR", 0.0),
            ("CO2_price_EUR_per_t", float(cfg.get("costs", {}).get("co2_price_eur_per_t", 0.0))),
            ("Include_CO2_in_objective", bool(cfg.get("costs", {}).get("include_co2_cost_in_objective", True))),
            ("Demand_charge_cost_EUR", 0.0),
            ("Period_fraction_of_year", float(n * dt_h / 8760.0) if n else 0.0),
            ("Objective_residual_EUR", 0.0),
        ]
    )

    grid_summary = OrderedDict(
        [
            ("Energy_from_grid_MWh", 0.0),
            ("Energy_to_grid_MWh", 0.0),
            ("Net_grid_import_MWh", 0.0),
            ("Average_purchase_price_EUR_MWh", 0.0),
            ("Average_sell_price_EUR_MWh", 0.0),
            ("Heat_dumped_MWh", 0.0),
            ("Dump_cost_rate_EUR_MWh", float(cfg.get("costs", {}).get("dump_cost_eur_per_mwh_th", 0.0))),
            ("Grid_CO2_emissions_t", 0.0),
            ("Total_CO2_emissions_t", 0.0),
        ]
    )

    summary_sections: OrderedDict[str, OrderedDict[str, Any]] = OrderedDict()
    summary_sections["objective"] = objective
    summary_sections["grid"] = grid_summary

    storage_section: OrderedDict[str, Any] | None = None
    if meta["storage"]:
        storage_key = f"storage_{meta['storage']['name']}"
        storage_section = OrderedDict(
            [
                ("Charge_MWh", 0.0),
                ("Discharge_MWh", 0.0),
                ("Average_SOC_MWh", 0.0),
                ("Min_SOC_MWh", 0.0),
                ("Max_SOC_MWh", 0.0),
                ("Capacity_MWh", meta["storage"]["e_max"]),
                ("Power_limit_MW", meta["storage"]["p_max"]),
            ]
        )
        summary_sections[storage_key] = storage_section

    heat_pump_sections: OrderedDict[str, OrderedDict[str, Any]] = OrderedDict()
    generator_sections: OrderedDict[str, OrderedDict[str, Any]] = OrderedDict()
    p2h_section: OrderedDict[str, Any] | None = None

    if model is not None and HAVE_PYOMO:
        times = list(model.t)

        def _extract(var: Any | None, key: str) -> None:
            if var is None:
                return
            try:
                series[key] = _extract_pyomo_series(var, times, key)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Fehler beim Verarbeiten der Serie %s: %s", key, exc)


        _extract(getattr(model, "P_buy", None), "P_buy_MW")
        _extract(getattr(model, "P_sell", None), "P_sell_MW")
        _extract(getattr(model, "Q_dump", None), "Q_dump_MWth")

        for hp in meta["heat_pumps"]:
            comp = hp["id"]
            _extract(getattr(model, f"{comp}_Q", None), f"{comp}_Q_th_MW")
            _extract(getattr(model, f"{comp}_Pel", None), f"{comp}_Pel_MW")
            _extract(getattr(model, f"{comp}_on", None), f"{comp}_on")

        if meta["storage"]:
            _extract(getattr(model, "TES_E", None), "TES_SOC_MWh")
            _extract(getattr(model, "TES_Qc", None), "TES_charge_MW")
            _extract(getattr(model, "TES_Qd", None), "TES_discharge_MW")

        for gen in meta["generators"]:
            comp = gen["name"]
            _extract(getattr(model, f"{comp}_Qth", None), f"{comp}_Q_th_MW")
            _extract(getattr(model, f"{comp}_fuel", None), f"{comp}_fuel_MW")
            if gen["has_el"]:
                _extract(getattr(model, f"{comp}_Pel", None), f"{comp}_Pel_MW")

        if meta["p2h"]:
            _extract(getattr(model, "P2H_Qth", None), "P2H_Q_th_MW")
            _extract(getattr(model, "P2H_Pel", None), "P2H_Pel_MW")

        objective["OBJ_value_EUR"] = float(pyo.value(model.obj)) if hasattr(model, "obj") else 0.0
        objective["P_buy_peak_MW"] = float(pyo.value(model.P_buy_peak)) if hasattr(model, "P_buy_peak") else 0.0
    else:
        times = list(range(1, n + 1))
        objective["OBJ_value_EUR"] = 0.0
        objective["P_buy_peak_MW"] = max(series["P_buy_MW"], default=0.0)

    price_series = table.data.get("strompreis_EUR_MWh", [0.0] * n)
    grid_co2_series = table.data.get("grid_co2_kg_MWh", [0.0] * n)
    demand_series = table.data.get("waermebedarf_MWth", [0.0] * n)

    include_gridcost = bool(cfg.get("costs", {}).get("include_gridcost_in_energy", False))
    energy_fee = float(cfg.get("grid", {}).get("energy_fee_eur_mwh", 0.0)) if include_gridcost else 0.0
    include_co2 = bool(cfg.get("costs", {}).get("include_co2_cost_in_objective", True))
    dump_cost_rate = float(cfg.get("costs", {}).get("dump_cost_eur_per_mwh_th", 0.0))
    include_demand = bool(cfg.get("costs", {}).get("include_demand_charge_in_rh", True))
    demand_charge_rate = float(cfg.get("grid", {}).get("demand_charge_eur_per_mw_y", 0.0))

    pbuy_series = series["P_buy_MW"]
    psell_series = series["P_sell_MW"]
    qdump_series = series["Q_dump_MWth"]

    energy_in = float(sum(pbuy_series) * dt_h)
    energy_out = float(sum(psell_series) * dt_h)
    heat_dump = float(sum(qdump_series) * dt_h)

    energy_cost = float(sum((pbuy_series[i] * (price_series[i] + energy_fee) * dt_h) for i in range(n))) if n else 0.0
    energy_revenue = float(sum((psell_series[i] * price_series[i] * dt_h) for i in range(n))) if n else 0.0
    grid_co2_t = float(sum((pbuy_series[i] * grid_co2_series[i] * dt_h) for i in range(n)) / 1000.0) if n else 0.0

    heat_demand_mwh = float(sum(demand_series) * dt_h) if n else 0.0

    fuel_cost_total = 0.0
    fuel_emissions_t = 0.0

    for hp in meta["heat_pumps"]:
        comp = hp["id"]
        heat_series = series[f"{comp}_Q_th_MW"]
        pel_series = series[f"{comp}_Pel_MW"]
        on_series = series[f"{comp}_on"]
        heat_mwh = float(sum(heat_series) * dt_h)
        pel_mwh = float(sum(pel_series) * dt_h)
        on_hours = float(sum(on_series) * dt_h)
        full_load = float((heat_mwh / hp["max_th"]) if hp["max_th"] > 0 else 0.0)
        heat_pump_sections[f"heat_pump_{comp}"] = OrderedDict(
            [
                ("Heat_output_MWh", heat_mwh),
                ("Electricity_input_MWh", pel_mwh),
                ("Operating_hours_h", on_hours),
                ("Full_load_hours_h", full_load),
                ("Thermal_capacity_MW", hp["max_th"]),
            ]
        )

    for gen in meta["generators"]:
        comp = gen["name"]
        heat_series = series[f"{comp}_Q_th_MW"]
        fuel_series = series[f"{comp}_fuel_MW"]
        heat_mwh = float(sum(heat_series) * dt_h)
        fuel_mwh = float(sum(fuel_series) * dt_h)
        pel_mwh = float(sum(series.get(f"{comp}_Pel_MW", [0.0] * n)) * dt_h) if gen["has_el"] else 0.0
        emission_t = float(fuel_mwh * gen["fuel_emission"] / 1000.0)
        cost_eur = float(fuel_mwh * gen["fuel_price"])
        entry = OrderedDict(
            [
                ("Heat_output_MWh", heat_mwh),
                ("Fuel_input_MWh", fuel_mwh),
                ("Fuel_cost_EUR", cost_eur),
                ("Fuel_price_EUR_MWh", gen["fuel_price"]),
                ("Fuel_emissions_t", emission_t),
                ("Fuel_bus", gen["fuel_bus"]),
                ("Thermal_capacity_MW", gen["cap_th"]),
            ]
        )
        if gen["has_el"]:
            entry["Power_output_MWh"] = float(pel_mwh)
        generator_sections[f"generator_{comp}"] = entry
        fuel_cost_total += cost_eur
        fuel_emissions_t += emission_t

    if meta["p2h"]:
        comp = meta["p2h"]["name"]
        heat_series = series[f"{comp}_Q_th_MW"]
        pel_series = series[f"{comp}_Pel_MW"]
        heat_mwh = float(sum(heat_series) * dt_h)
        pel_mwh = float(sum(pel_series) * dt_h)
        p2h_section = OrderedDict(
            [
                ("Heat_output_MWh", heat_mwh),
                ("Electricity_input_MWh", pel_mwh),
                ("Thermal_capacity_MW", meta["p2h"]["cap_th"]),
            ]
        )
        if meta["p2h"]["eff"]:
            p2h_section["Configured_efficiency"] = meta["p2h"]["eff"]

    if storage_section and meta["storage"]:
        charge_series = series["TES_charge_MW"]
        discharge_series = series["TES_discharge_MW"]
        soc_series = series["TES_SOC_MWh"]
        storage_section["Charge_MWh"] = float(sum(charge_series) * dt_h)
        storage_section["Discharge_MWh"] = float(sum(discharge_series) * dt_h)
        storage_section["Average_SOC_MWh"] = float(sum(soc_series) / len(soc_series)) if n else 0.0
        storage_section["Min_SOC_MWh"] = float(min(soc_series)) if soc_series else 0.0
        storage_section["Max_SOC_MWh"] = float(max(soc_series)) if soc_series else 0.0

    total_emissions_t = float(grid_co2_t + fuel_emissions_t)
    co2_price = float(cfg.get("costs", {}).get("co2_price_eur_per_t", 0.0))
    co2_cost = float(co2_price * total_emissions_t) if include_co2 else 0.0
    dump_cost = float(dump_cost_rate * heat_dump)

    demand_cost = 0.0
    if include_demand:
        if model is not None and HAVE_PYOMO and hasattr(model, "P_buy_peak"):
            peak = float(pyo.value(model.P_buy_peak))
        else:
            peak = float(max(pbuy_series, default=0.0))
        objective["P_buy_peak_MW"] = peak
        demand_cost = float(demand_charge_rate * objective["Period_fraction_of_year"] * peak)
    else:
        objective["P_buy_peak_MW"] = float(max(pbuy_series, default=0.0))

    objective["Grid_energy_cost_EUR"] = energy_cost
    objective["Grid_sell_revenue_EUR"] = energy_revenue
    objective["Grid_net_cost_EUR"] = energy_cost - energy_revenue
    objective["Fuel_cost_EUR"] = fuel_cost_total
    objective["Fuel_emissions_t"] = fuel_emissions_t
    objective["Dump_cost_EUR"] = dump_cost
    objective["CO2_cost_EUR"] = co2_cost
    objective["Demand_charge_cost_EUR"] = demand_cost

    components_sum = energy_cost - energy_revenue + fuel_cost_total + dump_cost + co2_cost + demand_cost
    objective["Objective_residual_EUR"] = objective["OBJ_value_EUR"] - components_sum

    grid_summary["Energy_from_grid_MWh"] = energy_in
    grid_summary["Energy_to_grid_MWh"] = energy_out
    grid_summary["Net_grid_import_MWh"] = energy_in - energy_out
    grid_summary["Average_purchase_price_EUR_MWh"] = float(energy_cost / energy_in) if energy_in else 0.0
    grid_summary["Average_sell_price_EUR_MWh"] = float(energy_revenue / energy_out) if energy_out else 0.0
    grid_summary["Heat_dumped_MWh"] = heat_dump
    grid_summary["Grid_CO2_emissions_t"] = grid_co2_t
    grid_summary["Total_CO2_emissions_t"] = total_emissions_t
    grid_summary["Heat_demand_MWh"] = heat_demand_mwh

    for name, section in heat_pump_sections.items():
        summary_sections[name] = section
    for name, section in generator_sections.items():
        summary_sections[name] = section
    if p2h_section:
        summary_sections["p2h"] = p2h_section

    flat = _flatten_summary(summary_sections)
    return series, summary_sections, flat
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

    solver_result = None
    solver_requested = run_cfg.get("solver", "glpk")
    solver_used = solver_requested
    model_for_summary = m if HAVE_PYOMO and m is not None else None

    if model_for_summary is not None:
        try:
            opt = pyo.SolverFactory(solver_requested)
        except Exception:
            opt = pyo.SolverFactory("glpk")
            solver_used = "glpk"
        else:
            solver_used = solver_requested
        solver_result = opt.solve(model_for_summary, tee=False)
    series, summary_sections, costs = _collect_timeseries_and_summary(table, cfg, dt_h, model_for_summary)

    solver_meta = OrderedDict()
    solver_meta["solver_requested"] = solver_requested
    solver_meta["solver_used"] = solver_used
    solver_meta["pyomo_available"] = HAVE_PYOMO
    solver_meta["model_built"] = m is not None
    if solver_result is not None:
        solver_info = getattr(solver_result, "solver", None)
        solver_meta["status"] = str(getattr(solver_info, "status", "unknown")) if solver_info is not None else "unknown"
        solver_meta["termination_condition"] = (
            str(getattr(solver_info, "termination_condition", "unknown")) if solver_info is not None else "unknown"
        )
        if solver_info is not None:
            solver_meta["wallclock_time_s"] = getattr(solver_info, "wallclock_time", None)
            solver_meta["user_time_s"] = getattr(solver_info, "user_time", None)
            solver_meta["iterations"] = getattr(solver_info, "iterations", None)
            solver_meta["return_code"] = getattr(solver_info, "return_code", None)
            message = getattr(solver_info, "message", None)
            if message:
                solver_meta["message"] = str(message)
    else:
        solver_meta["status"] = "not_run"
        solver_meta["termination_condition"] = None

    stamp = time.strftime("%Y%m%d_%H%M%S")
    mode = str(scenario_cfg.get("mode", "PF"))
    title = str(scenario_cfg.get("title", "Baseline"))
    tag = scenario_cfg.get("tag") or f"{mode}-{title}"
    outdir = os.path.join("exports", f"{stamp}_{_slugify(tag)}")
    os.makedirs(outdir, exist_ok=True)

    scen_file = os.path.join(outdir, "scenario_timeseries.csv")
    write_timeseries_csv(scen_file, table, series)

    metadata_sections: OrderedDict[str, OrderedDict[str, Any]] = OrderedDict()
    metadata_sections["run"] = OrderedDict(
        [
            ("timestamp", stamp),
            ("output_directory", outdir),
            ("dt_h", dt_h),
            ("n_steps", len(table)),
            ("period_hours", len(table) * dt_h),
            ("pyomo_available", HAVE_PYOMO),
        ]
    )
    if isinstance(scenario_cfg, dict):
        metadata_sections["scenario"] = OrderedDict((key, value) for key, value in scenario_cfg.items())
    else:
        metadata_sections["scenario"] = OrderedDict(value=scenario_cfg)
    if isinstance(site, dict):
        metadata_sections["site"] = OrderedDict((key, value) for key, value in site.items())
    else:
        metadata_sections["site"] = OrderedDict(value=site)
    metadata_sections["solver"] = solver_meta

    xlsx_file = os.path.join(outdir, "scenario_results.xlsx")
    try:
        write_excel_workbook(xlsx_file, table, series, summary_sections, metadata_sections)
    except RuntimeError as exc:
        print(f"[EXPORT] Excel-Export übersprungen: {exc}")
        xlsx_file = None

    summary_json = _json_safe({section: dict(metrics) for section, metrics in summary_sections.items()})
    metadata_json = _json_safe({section: dict(entries) for section, entries in metadata_sections.items()})

    with open(os.path.join(outdir, "costs.json"), "w", encoding="utf-8") as handle:
        json.dump(_json_safe(costs), handle, indent=2)

    with open(os.path.join(outdir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary_json, handle, indent=2)

    with open(os.path.join(outdir, "metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata_json, handle, indent=2)

    with open(os.path.join(outdir, "merged_config.json"), "w", encoding="utf-8") as handle:
        json.dump(_json_safe(cfg), handle, indent=2)

    return {
        "scenario_csv": scen_file,
        "scenario_xlsx": xlsx_file,
        "outdir": outdir,
        "costs": costs,
        "summary": summary_json,
        "metadata": metadata_json,
        "scenario": scenario_cfg,
    }

def _extract_pyomo_series(var: Any, times: Sequence[Any], name: str, tolerance: float = 1e-9) -> List[float]:
    values: List[float] = []
    for t in times:
        try:
            raw = pyo.value(var[t])
        except Exception as exc:  # pragma: no cover - requires Pyomo specific failures
            logger.warning("Konnte Pyomo-Wert %s[%s] nicht auslesen: %s", name, t, exc)
            values.append(0.0)
            continue

        if raw is None:
            logger.warning("Pyomo-Wert %s[%s] ist None", name, t)
            values.append(0.0)
            continue

        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            logger.warning(
                "Pyomo-Wert %s[%s] kann nicht in float umgewandelt werden: %s", name, t, exc
            )
            values.append(0.0)
            continue

        if not math.isfinite(number):
            logger.warning("Pyomo-Wert %s[%s] ist nicht endlich (%s)", name, t, number)
            values.append(0.0)
            continue

        if abs(number) < tolerance:
            number = 0.0

        values.append(number)

    return values

