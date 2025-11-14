"""Workflow helpers for rolling horizon simulations.

The module implements a lightweight orchestration layer around the existing
`energis.run.orchestrator` utilities.  Callers can describe a sequence of
workflow steps via ``scenario.workflow`` (e.g. ``["PF", "RH"]``) or the legacy
``run_mode`` switch.  This makes it trivial to compare PF-only, RH-only and
combined PF→RH simulations while reusing the same configuration set.  The RH
logic honours configuration entries for window size, step width and terminal
policies while preserving state-of-charge (SOC) values between windows.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import argparse
import copy
import math
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

try:  # pragma: no cover - optional dependency
    import pyomo.environ as pyo
    HAVE_PYOMO = True
except Exception:  # pragma: no cover - the solver stack is optional for tests
    HAVE_PYOMO = False
    pyo = None

from energis.config.merge import load_and_merge
from energis.io.loader import load_input_excel
from energis.models.system_builder import build_model
from energis.utils.timeseries import TimeSeriesTable
from . import orchestrator


@dataclass
class ScenarioResult:
    """Container for a single optimisation run."""

    table: TimeSeriesTable
    series: OrderedDict[str, List[float]]
    summary: Mapping[str, Mapping[str, Any]]
    costs: Dict[str, Any]
    solver: Dict[str, Any]


@dataclass
class WindowResult(ScenarioResult):
    """Specialised result carrying metadata for RH windows."""

    start_index: int
    commit_steps: int


@dataclass
class RollingHorizonResult:
    """Aggregated result for a complete RH simulation."""

    table: TimeSeriesTable
    series: OrderedDict[str, List[float]]
    costs: Dict[str, Any]
    windows: List[WindowResult]


@dataclass
class DesignData:
    """Design figures extracted from a PF optimisation."""

    heat_pumps: Dict[str, Dict[str, float]]
    storage: Optional[Dict[str, float]]


@dataclass
class WorkflowPlan:
    """Parsed representation of the requested workflow sequence."""

    steps: Sequence[str]
    fix_design: bool


@dataclass
class _RollingParams:
    """Internal helper capturing rolling horizon parameters."""

    horizon_hours: float
    step_hours: float
    terminal_policy: str

    def as_steps(self, dt_h: float) -> tuple[int, int]:
        """Return window and step lengths measured in simulation steps."""

        if dt_h <= 0:
            raise ValueError("dt_h must be positive")
        horizon_steps = _hours_to_steps(self.horizon_hours, dt_h, "HEAT_HORIZON_HOURS")
        step_steps = _hours_to_steps(self.step_hours, dt_h, "STEP_HOURS")
        if step_steps > horizon_steps:
            raise ValueError("STEP_HOURS must not exceed HEAT_HORIZON_HOURS")
        return horizon_steps, step_steps


@dataclass
class WorkflowContext:
    """Mutable state shared between workflow steps."""

    cfg: Dict[str, Any]
    table: TimeSeriesTable
    dt_h: float
    solver_name: str
    plan: WorkflowPlan
    pf_result: Optional[ScenarioResult] = None
    rh_result: Optional[RollingHorizonResult] = None
    design: Optional[DesignData] = None


StepHandler = Callable[[WorkflowContext], None]


_STEP_HANDLERS: Dict[str, StepHandler] = {}


def register_workflow_step(name: str, handler: StepHandler) -> None:
    """Register or replace a workflow step handler.

    Parameters
    ----------
    name:
        Identifier used in ``scenario.workflow`` entries.  The identifier is
        stored in upper-case to provide case-insensitive matching.
    handler:
        Callable that mutates a :class:`WorkflowContext` in place.
    """

    key = str(name).strip().upper()
    if not key:
        raise ValueError("Workflow step name must not be empty")
    _STEP_HANDLERS[key] = handler


def unregister_workflow_step(name: str) -> None:
    """Remove a workflow step handler if it exists."""

    key = str(name).strip().upper()
    if key:
        _STEP_HANDLERS.pop(key, None)


def get_registered_workflow_steps() -> List[str]:
    """Return the currently registered workflow step identifiers."""

    return sorted(_STEP_HANDLERS.keys())


@dataclass
class WorkflowResult:
    """Return value for :func:`run_workflow`."""

    config: Dict[str, Any]
    pf_result: Optional[ScenarioResult]
    rh_result: Optional[RollingHorizonResult]
    design: Optional[DesignData]
    plan: WorkflowPlan


def run_workflow(config_paths: List[str], overrides: Optional[Dict[str, Any]] = None) -> WorkflowResult:
    """Execute the configured workflow (PF, RH or PF→RH).

    Parameters
    ----------
    config_paths:
        List of configuration file paths that should be merged.
    overrides:
        Optional dictionary applied on top of the merged configuration.
    """

    cfg = load_and_merge(config_paths)
    if overrides:
        cfg = orchestrator._deep_update(cfg, overrides)  # type: ignore[attr-defined]

    run_cfg = cfg.get("run", {})
    scenario_cfg = cfg.get("scenario", {})
    site_cfg = cfg.get("site", {})

    dt_h = float(run_cfg.get("dt_h", 1.0))
    table = load_input_excel(site_cfg.get("input_xlsx", "Import_Data.xlsx"), site_cfg, dt_hours=dt_h)
    table.ensure_frequency(dt_h)
    table = orchestrator._apply_horizon(table, scenario_cfg, dt_h)  # type: ignore[attr-defined]
    orchestrator._assert_capacity_vs_demand(table, cfg)  # type: ignore[attr-defined]

    plan = _parse_workflow_plan(scenario_cfg)
    solver_name = str(run_cfg.get("solver", "glpk"))

    context = WorkflowContext(cfg, table, dt_h, solver_name, plan)

    for step in plan.steps:
        handler = _STEP_HANDLERS.get(step)
        if handler is None:
            raise ValueError(f"Unsupported workflow step: {step}")
        handler(context)

    return WorkflowResult(cfg, context.pf_result, context.rh_result, context.design, plan)


def _parse_workflow_plan(scenario_cfg: Mapping[str, Any]) -> WorkflowPlan:
    run_mode = str(scenario_cfg.get("run_mode", "")).strip().upper() or "PF_ONLY"
    workflow = scenario_cfg.get("workflow")

    if workflow is not None:
        if isinstance(workflow, (str, bytes)):
            steps = [str(workflow)]
        else:
            steps = list(workflow)
        steps_upper = [str(step).strip().upper() for step in steps if str(step).strip()]
    else:
        mapping = {
            "PF_ONLY": ["PF"],
            "RH_ONLY": ["RH"],
            "PF_THEN_RH": ["PF", "RH"],
            "PF_AND_RH": ["PF", "RH"],
        }
        steps_upper = mapping.get(run_mode, ["PF"])

    if not steps_upper:
        raise ValueError("Workflow must contain at least one step")

    fix_default = run_mode in {"PF_THEN_RH", "PF_AND_RH"}
    fix_design = bool(scenario_cfg.get("fix_design", scenario_cfg.get("fix_design_in_rh", fix_default)))

    return WorkflowPlan(steps=steps_upper, fix_design=fix_design)


def _pf_step(context: WorkflowContext) -> None:
    result = _solve_scenario(context.table, context.cfg, context.dt_h, context.solver_name)
    context.pf_result = result
    context.design = _extract_design_data(result.summary)


def _rh_step(context: WorkflowContext) -> None:
    params = _load_rolling_params(context.cfg)
    horizon_steps, step_steps = params.as_steps(context.dt_h)
    fix_design = context.plan.fix_design and context.design is not None
    context.rh_result = _run_rolling_horizon(
        context.cfg,
        context.table,
        context.dt_h,
        context.solver_name,
        params,
        horizon_steps,
        step_steps,
        context.design,
        fix_design,
    )


def _run_rolling_horizon(
    base_cfg: Dict[str, Any],
    table: TimeSeriesTable,
    dt_h: float,
    solver_name: str,
    params: _RollingParams,
    horizon_steps: int,
    step_steps: int,
    design: Optional[DesignData],
    fix_design: bool,
) -> RollingHorizonResult:
    n = len(table)
    if n == 0:
        empty_series: OrderedDict[str, List[float]] = OrderedDict()
        return RollingHorizonResult(table, empty_series, {}, [])

    aggregated_indices: List[int] = []
    aggregated_series: OrderedDict[str, List[float]] = OrderedDict()
    windows: List[WindowResult] = []

    soc_next = _initial_soc(base_cfg)
    base_storage_enabled = _storage_enabled(base_cfg)

    start = 0
    while start < n:
        end = min(start + horizon_steps, n)
        indices = list(range(start, end))
        window_table = orchestrator._slice_table(table, indices)  # type: ignore[attr-defined]
        window_cfg = copy.deepcopy(base_cfg)

        if params.terminal_policy:
            _apply_terminal_policy(window_cfg, params.terminal_policy)
        if soc_next is not None and base_storage_enabled:
            _set_initial_soc(window_cfg, soc_next)
        if fix_design and design is not None:
            window_cfg = _apply_design_fix(window_cfg, design)

        window_result = _solve_scenario(window_table, window_cfg, dt_h, solver_name)
        commit_len = min(step_steps, len(window_table))

        _extend_series(aggregated_series, window_result.series, commit_len)
        aggregated_indices.extend(indices[:commit_len])

        soc_next = _next_soc(window_result.series, commit_len, soc_next)

        windows.append(
            WindowResult(
                table=window_result.table,
                series=window_result.series,
                summary=window_result.summary,
                costs=window_result.costs,
                solver=window_result.solver,
                start_index=start,
                commit_steps=commit_len,
            )
        )

        start += step_steps

    if aggregated_indices != list(range(n)):
        raise RuntimeError("Rolling horizon aggregation did not cover the full time series")

    aggregated_table = orchestrator._slice_table(table, aggregated_indices)  # type: ignore[attr-defined]
    aggregated_costs = _aggregate_cost_summary(
        aggregated_table,
        aggregated_series,
        base_cfg,
        dt_h,
        windows,
        design,
    )

    return RollingHorizonResult(aggregated_table, aggregated_series, aggregated_costs, windows)


def _extend_series(
    global_series: MutableMapping[str, List[float]],
    window_series: Mapping[str, List[float]],
    commit_len: int,
) -> None:
    if commit_len <= 0:
        return
    for col in list(global_series.keys()):
        if col not in window_series:
            global_series[col].extend([0.0] * commit_len)
    for col, values in window_series.items():
        dest = global_series.setdefault(col, [])
        slice_values = list(values[:commit_len])
        if len(slice_values) < commit_len:
            slice_values.extend([0.0] * (commit_len - len(slice_values)))
        dest.extend(slice_values)


def _next_soc(series: Mapping[str, List[float]], commit_len: int, fallback: Optional[float]) -> Optional[float]:
    soc_series = series.get("TES_SOC_MWh")
    if soc_series is None or commit_len <= 0:
        return fallback
    idx = min(commit_len - 1, len(soc_series) - 1)
    return float(soc_series[idx]) if idx >= 0 else fallback


def _solve_scenario(
    table: TimeSeriesTable,
    cfg: Dict[str, Any],
    dt_h: float,
    solver_name: str,
) -> ScenarioResult:
    model = build_model(table, cfg, dt_h=dt_h)
    solver_meta: Dict[str, Any] = {
        "solver_requested": solver_name,
        "pyomo_available": HAVE_PYOMO,
        "model_built": model is not None,
    }
    if model is not None and HAVE_PYOMO:
        solver_used = solver_name
        try:
            opt = pyo.SolverFactory(solver_name)
        except Exception:  # pragma: no cover - solver fallback
            solver_used = "glpk"
            opt = pyo.SolverFactory("glpk")
        solver_result = opt.solve(model, tee=False)
        solver_meta["solver_used"] = solver_used
        solver_meta["status"] = str(getattr(getattr(solver_result, "solver", None), "status", "unknown"))
        solver_meta["termination_condition"] = str(
            getattr(getattr(solver_result, "solver", None), "termination_condition", "unknown")
        )
    else:
        solver_meta["solver_used"] = solver_name
        solver_meta["status"] = "not_run"
        solver_meta["termination_condition"] = None

    series, summary, costs = orchestrator._collect_timeseries_and_summary(  # type: ignore[attr-defined]
        table,
        cfg,
        dt_h,
        model if HAVE_PYOMO else None,
    )
    return ScenarioResult(table, series, summary, costs, solver_meta)


def _extract_design_data(summary: Mapping[str, Mapping[str, Any]]) -> DesignData:
    heat_pumps: Dict[str, Dict[str, float]] = {}
    storage: Optional[Dict[str, float]] = None

    for key, metrics in summary.items():
        if key.startswith("heat_pump_"):
            hp_id = key.split("heat_pump_", 1)[1]
            capacity = float(metrics.get("Thermal_capacity_MW", 0.0))
            build = float(metrics.get("Build_binary", metrics.get("Build", 0.0)))
            heat_pumps[hp_id] = {
                "capacity_mw": capacity,
                "build_binary": build,
            }
        elif key.startswith("storage_"):
            storage = {
                "name": key.split("storage_", 1)[1],
                "capacity_mwh": float(metrics.get("Capacity_MWh", 0.0)),
                "power_mw": float(metrics.get("Power_limit_MW", 0.0)),
                "build_binary": float(metrics.get("Build_binary", metrics.get("Build", 0.0))),
            }

    return DesignData(heat_pumps=heat_pumps, storage=storage)


def _load_rolling_params(cfg: Mapping[str, Any]) -> _RollingParams:
    scenario_cfg = cfg.get("scenario", {}) if isinstance(cfg.get("scenario"), dict) else {}
    rolling_cfg = scenario_cfg.get("rolling_horizon") or cfg.get("rolling_horizon", {}) or {}

    def _get(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
        for key in keys:
            if key in mapping:
                return mapping[key]
        return default

    horizon_hours = float(_get(rolling_cfg, "HEAT_HORIZON_HOURS", "heat_horizon_hours", default=168.0))
    step_hours = float(_get(rolling_cfg, "STEP_HOURS", "step_hours", default=horizon_hours))
    terminal_policy = str(_get(rolling_cfg, "terminal_policy", "TERMINAL_POLICY", default="")).strip().lower()

    return _RollingParams(horizon_hours=horizon_hours, step_hours=step_hours, terminal_policy=terminal_policy)


def _hours_to_steps(hours: float, dt_h: float, name: str) -> int:
    if hours <= 0:
        raise ValueError(f"{name} must be positive")
    ratio = hours / dt_h
    rounded = round(ratio)
    if not math.isclose(ratio, rounded, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{name} must be a multiple of dt_h")
    return max(1, int(rounded))


def _initial_soc(cfg: Mapping[str, Any]) -> Optional[float]:
    system = cfg.get("system", {}) if isinstance(cfg.get("system"), dict) else {}
    storage = system.get("storage", {}) if isinstance(system.get("storage"), dict) else {}
    if not storage or not storage.get("enabled", False):
        return None
    inputs = cfg.get("inputs", {}) if isinstance(cfg.get("inputs"), dict) else {}
    if "SOC_init" in inputs:
        return float(inputs["SOC_init"])
    if "SOC_init" in storage:
        return float(storage["SOC_init"])
    if "soc0_mwh" in storage:
        return float(storage["soc0_mwh"])
    return 0.0


def _storage_enabled(cfg: Mapping[str, Any]) -> bool:
    system = cfg.get("system", {}) if isinstance(cfg.get("system"), dict) else {}
    storage = system.get("storage", {}) if isinstance(system.get("storage"), dict) else {}
    return bool(storage.get("enabled", False))


def _set_initial_soc(cfg: MutableMapping[str, Any], soc: float) -> None:
    inputs = cfg.setdefault("inputs", {})
    if isinstance(inputs, dict):
        inputs["SOC_init"] = float(soc)
    storage = cfg.setdefault("system", {}).setdefault("storage", {})
    if isinstance(storage, dict):
        storage["soc0_mwh"] = float(soc)


def _apply_terminal_policy(cfg: MutableMapping[str, Any], policy: str) -> None:
    if not policy:
        return
    system = cfg.setdefault("system", {})
    if not isinstance(system, dict):
        return
    storage = system.setdefault("storage", {})
    if not isinstance(storage, dict):
        return
    terminal = storage.setdefault("terminal", {})
    if isinstance(terminal, dict):
        terminal["policy"] = policy


def _apply_design_fix(cfg: Dict[str, Any], design: DesignData) -> Dict[str, Any]:
    cfg_copy = copy.deepcopy(cfg)
    system = cfg_copy.setdefault("system", {})
    heat_pumps = system.get("heat_pumps")
    if isinstance(heat_pumps, list):
        for hp_cfg in heat_pumps:
            if not isinstance(hp_cfg, dict):
                continue
            hp_id = str(hp_cfg.get("id"))
            if hp_id not in design.heat_pumps:
                continue
            design_entry = design.heat_pumps[hp_id]
            capacity = float(design_entry.get("capacity_mw", 0.0))
            build_binary = float(design_entry.get("build_binary", 0.0))
            invest_cfg = hp_cfg.setdefault("investment", {})
            if isinstance(invest_cfg, dict):
                invest_cfg["enabled"] = False
                invest_cfg["capacity_min_mw"] = capacity
                invest_cfg["capacity_max_mw"] = capacity
            hp_cfg["max_th_mw"] = capacity
            hp_cfg["min_th_mw"] = capacity
            if build_binary < 0.5:
                hp_cfg["enabled"] = False

    storage_cfg = system.get("storage") if isinstance(system.get("storage"), dict) else None
    if storage_cfg and design.storage:
        storage_cfg["enabled"] = bool(design.storage.get("build_binary", 0.0) >= 0.5)
        storage_cfg["max_energy_mwh"] = float(design.storage.get("capacity_mwh", 0.0))
        storage_cfg["max_power_mw"] = float(design.storage.get("power_mw", 0.0))
        invest_cfg = storage_cfg.setdefault("investment", {})
        if isinstance(invest_cfg, dict):
            invest_cfg["enabled"] = False
            invest_cfg["energy_capacity_min_mwh"] = float(design.storage.get("capacity_mwh", 0.0))
            invest_cfg["energy_capacity_max_mwh"] = float(design.storage.get("capacity_mwh", 0.0))
            invest_cfg["power_capacity_min_mw"] = float(design.storage.get("power_mw", 0.0))
            invest_cfg["power_capacity_max_mw"] = float(design.storage.get("power_mw", 0.0))

    return cfg_copy


def _aggregate_cost_summary(
    table: TimeSeriesTable,
    series_map: Mapping[str, Sequence[float]],
    cfg: Mapping[str, Any],
    dt_h: float,
    windows: Sequence[WindowResult],
    design: Optional[DesignData],
) -> Dict[str, Any]:
    """Reconstruct a flattened cost/summary dictionary for committed RH slices."""

    n = len(table)
    if n == 0:
        return {}

    meta = copy.deepcopy(orchestrator._gather_component_metadata(cfg))  # type: ignore[attr-defined]
    _apply_design_to_metadata(meta, design)

    def _series(name: str) -> List[float]:
        values = list(series_map.get(name, []))
        if len(values) < n:
            values.extend([0.0] * (n - len(values)))
        elif len(values) > n:
            values = values[:n]
        return [float(v) for v in values]

    grid_cfg = cfg.get("grid", {}) if isinstance(cfg.get("grid"), dict) else {}
    costs_cfg = cfg.get("costs", {}) if isinstance(cfg.get("costs"), dict) else {}

    period_fraction = float(n * dt_h / 8760.0)
    demand_year_fraction = float(grid_cfg.get("year_fraction", period_fraction))

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
            ("CO2_price_EUR_per_t", float(costs_cfg.get("co2_price_eur_per_t", 0.0))),
            (
                "Include_CO2_in_objective",
                bool(costs_cfg.get("include_co2_cost_in_objective", True)),
            ),
            ("Demand_charge_cost_EUR", 0.0),
            ("Capex_cost_EUR", 0.0),
            ("Activation_cost_EUR", 0.0),
            ("Tie_breaker_cost_EUR", 0.0),
            ("Storage_installation_cost_EUR", 0.0),
            ("Period_fraction_of_year", period_fraction),
            ("Demand_charge_year_fraction", demand_year_fraction),
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
            ("Dump_cost_rate_EUR_MWh", float(costs_cfg.get("dump_cost_eur_per_mwh_th", 0.0))),
            ("Grid_CO2_emissions_t", 0.0),
            ("Total_CO2_emissions_t", 0.0),
            ("Heat_demand_MWh", 0.0),
        ]
    )

    summary_sections: OrderedDict[str, OrderedDict[str, Any]] = OrderedDict()
    summary_sections["objective"] = objective
    summary_sections["grid"] = grid_summary

    storage_section: Optional[OrderedDict[str, Any]] = None
    if meta.get("storage"):
        storage = meta["storage"]
        storage_section = OrderedDict(
            [
                ("Charge_MWh", 0.0),
                ("Discharge_MWh", 0.0),
                ("Average_SOC_MWh", 0.0),
                ("Min_SOC_MWh", 0.0),
                ("Max_SOC_MWh", 0.0),
                ("Capacity_MWh", float(storage.get("e_cap_init", storage.get("e_max", 0.0)))),
                ("Power_limit_MW", float(storage.get("p_cap_init", storage.get("p_max", 0.0)))),
                ("Build_binary", 1.0 if storage.get("e_cap_init", storage.get("e_max", 0.0)) > 0 else 0.0),
                ("Investment_enabled", bool(storage.get("invest_enabled", False))),
            ]
        )
        storage_section["Capacity_bounds_MWh"] = [
            storage.get("e_cap_min", 0.0),
            storage.get("e_cap_max", storage_section["Capacity_MWh"]),
        ]
        storage_section["Power_bounds_MW"] = [
            storage.get("p_cap_min", 0.0),
            storage.get("p_cap_max", storage_section["Power_limit_MW"]),
        ]
        summary_sections[f"storage_{storage.get('name', 'TES')}"] = storage_section

    heat_pump_sections: OrderedDict[str, OrderedDict[str, Any]] = OrderedDict()
    generator_sections: OrderedDict[str, OrderedDict[str, Any]] = OrderedDict()
    p2h_section: Optional[OrderedDict[str, Any]] = None

    p_buy = _series("P_buy_MW")
    p_sell = _series("P_sell_MW")
    q_dump = _series("Q_dump_MWth")

    price_series = table.data.get("strompreis_EUR_MWh", [0.0] * n)
    grid_co2_series = table.data.get("grid_co2_kg_MWh", [0.0] * n)
    demand_series = table.data.get("waermebedarf_MWth", [0.0] * n)

    include_gridcost = bool(costs_cfg.get("include_gridcost_in_energy", False))
    energy_fee = float(grid_cfg.get("energy_fee_eur_mwh", 0.0))
    grid_cost = float(grid_cfg.get("gridcost_eur_mwh", 0.0))
    include_demand = bool(
        grid_cfg.get(
            "include_demand_charge_in_rh",
            costs_cfg.get("include_demand_charge_in_rh", True),
        )
    )
    demand_charge_rate = float(grid_cfg.get("demand_charge_eur_per_mw_y", 0.0))

    addition = (energy_fee + grid_cost) if include_gridcost else 0.0

    buy_prices = [float(price_series[i]) + addition for i in range(n)] if n else []
    sell_prices = [float(price_series[i]) for i in range(n)] if n else []

    def _sell_price(base: float) -> float:
        sell_floor = float(grid_cfg.get("sell_floor_eur_mwh", 0.0))
        sell_haircut = float(grid_cfg.get("sell_haircut_fraction", 0.0))
        sell_spread = float(grid_cfg.get("sell_spread_eur_mwh", 0.0))
        sell_fee = float(grid_cfg.get("sell_fee_eur_mwh", 0.0))
        sell_premium = float(grid_cfg.get("sell_premium_eur_mwh", 0.0))
        price = max(base - sell_spread, sell_floor)
        price = price * max(0.0, 1.0 - sell_haircut)
        price = price - sell_fee + sell_premium
        return max(price, 0.0)

    sell_prices = [_sell_price(price) for price in sell_prices]

    energy_in = float(sum(p_buy) * dt_h)
    energy_out = float(sum(p_sell) * dt_h)
    heat_dump = float(sum(q_dump) * dt_h)

    energy_cost = float(sum(p_buy[i] * buy_prices[i] * dt_h for i in range(n))) if n else 0.0
    energy_revenue = float(sum(p_sell[i] * sell_prices[i] * dt_h for i in range(n))) if n else 0.0
    grid_co2_t = float(sum(p_buy[i] * float(grid_co2_series[i]) * dt_h for i in range(n)) / 1000.0) if n else 0.0

    heat_demand_mwh = float(sum(float(demand_series[i]) for i in range(n)) * dt_h) if n else 0.0

    fuel_cost_total = 0.0
    fuel_emissions_t = 0.0

    for hp in meta.get("heat_pumps", []):
        comp = hp["id"]
        heat_series = _series(f"{comp}_Q_th_MW")
        pel_series = _series(f"{comp}_Pel_MW")
        on_series = _series(f"{comp}_on")

        heat_mwh = float(sum(heat_series) * dt_h)
        pel_mwh = float(sum(pel_series) * dt_h)
        on_hours = float(sum(on_series) * dt_h)

        cap_value = float(hp.get("cap_init", hp.get("max_th", 0.0)))
        full_load = float((heat_mwh / cap_value) if cap_value > 1e-9 else 0.0)
        avg_cop = float((heat_mwh / pel_mwh) if pel_mwh > 1e-9 else 0.0)
        build_value = 1.0 if cap_value > 0 else 0.0

        heat_pump_sections[f"heat_pump_{comp}"] = OrderedDict(
            [
                ("Heat_output_MWh", heat_mwh),
                ("Electricity_input_MWh", pel_mwh),
                ("Operating_hours_h", on_hours),
                ("Full_load_hours_h", full_load),
                ("Thermal_capacity_MW", cap_value),
                ("Build_binary", build_value),
                ("Investment_enabled", bool(hp.get("invest_enabled", False))),
                (
                    "Capacity_bounds_MW",
                    [hp.get("cap_min", 0.0), hp.get("cap_max", cap_value)],
                ),
                ("Average_COP", avg_cop),
            ]
        )

    for gen in meta.get("generators", []):
        comp = gen["name"]
        heat_series = _series(f"{comp}_Q_th_MW")
        fuel_series = _series(f"{comp}_fuel_MW")
        pel_series = _series(f"{comp}_Pel_MW") if gen.get("has_el") else [0.0] * n

        heat_mwh = float(sum(heat_series) * dt_h)
        fuel_mwh = float(sum(fuel_series) * dt_h)
        pel_mwh = float(sum(pel_series) * dt_h) if gen.get("has_el") else 0.0

        fuel_price = float(gen.get("fuel_price", 0.0))
        fuel_emission = float(gen.get("fuel_emission", 0.0))

        cost_eur = float(fuel_mwh * fuel_price)
        emission_t = float(fuel_mwh * fuel_emission / 1000.0)

        entry = OrderedDict(
            [
                ("Heat_output_MWh", heat_mwh),
                ("Fuel_input_MWh", fuel_mwh),
                ("Fuel_cost_EUR", cost_eur),
                ("Fuel_price_EUR_MWh", fuel_price),
                ("Fuel_emissions_t", emission_t),
                ("Fuel_bus", gen.get("fuel_bus", "")),
                ("Thermal_capacity_MW", gen.get("cap_th", 0.0)),
            ]
        )
        if gen.get("has_el"):
            entry["Power_output_MWh"] = float(pel_mwh)

        generator_sections[f"generator_{comp}"] = entry
        fuel_cost_total += cost_eur
        fuel_emissions_t += emission_t

    if meta.get("p2h"):
        comp = meta["p2h"]["name"]
        heat_series = _series(f"{comp}_Q_th_MW")
        pel_series = _series(f"{comp}_Pel_MW")
        heat_mwh = float(sum(heat_series) * dt_h)
        pel_mwh = float(sum(pel_series) * dt_h)
        p2h_section = OrderedDict(
            [
                ("Heat_output_MWh", heat_mwh),
                ("Electricity_input_MWh", pel_mwh),
                ("Thermal_capacity_MW", meta["p2h"].get("cap_th", 0.0)),
            ]
        )
        eff = meta["p2h"].get("eff")
        if eff:
            p2h_section["Configured_efficiency"] = eff

    if storage_section is not None:
        charge_series = _series("TES_charge_MW")
        discharge_series = _series("TES_discharge_MW")
        soc_series = _series("TES_SOC_MWh")

        storage_section["Charge_MWh"] = float(sum(charge_series) * dt_h)
        storage_section["Discharge_MWh"] = float(sum(discharge_series) * dt_h)
        storage_section["Average_SOC_MWh"] = float(sum(soc_series) / len(soc_series)) if soc_series else 0.0
        storage_section["Min_SOC_MWh"] = float(min(soc_series)) if soc_series else 0.0
        storage_section["Max_SOC_MWh"] = float(max(soc_series)) if soc_series else 0.0

    for name, section in heat_pump_sections.items():
        summary_sections[name] = section
    for name, section in generator_sections.items():
        summary_sections[name] = section
    if p2h_section:
        summary_sections["p2h"] = p2h_section

    extras = {
        "Capex_cost_EUR": 0.0,
        "Activation_cost_EUR": 0.0,
        "Tie_breaker_cost_EUR": 0.0,
        "Storage_installation_cost_EUR": 0.0,
    }
    for window in windows:
        for key in extras:
            cost_key = f"objective.{key}"
            if cost_key in window.costs:
                extras[key] += float(window.costs[cost_key])

    include_co2 = bool(objective["Include_CO2_in_objective"])
    co2_price = float(objective["CO2_price_EUR_per_t"])

    total_emissions_t = float(grid_co2_t + fuel_emissions_t)
    co2_cost = float(co2_price * total_emissions_t) if include_co2 else 0.0
    dump_cost_rate = float(costs_cfg.get("dump_cost_eur_per_mwh_th", 0.0))
    dump_cost = float(dump_cost_rate * heat_dump)

    if include_demand:
        peak = float(max(p_buy, default=0.0))
        demand_cost = float(demand_charge_rate * demand_year_fraction * peak)
    else:
        peak = float(max(p_buy, default=0.0))
        demand_cost = 0.0

    energy_net_cost = float(energy_cost - energy_revenue)

    objective["P_buy_peak_MW"] = peak
    objective["Grid_energy_cost_EUR"] = energy_cost
    objective["Grid_sell_revenue_EUR"] = energy_revenue
    objective["Grid_net_cost_EUR"] = energy_net_cost
    objective["Fuel_cost_EUR"] = fuel_cost_total
    objective["Fuel_emissions_t"] = fuel_emissions_t
    objective["Dump_cost_EUR"] = dump_cost
    objective["CO2_cost_EUR"] = co2_cost
    objective["Demand_charge_cost_EUR"] = demand_cost

    for key, value in extras.items():
        objective[key] = value

    components_sum = (
        energy_net_cost
        + fuel_cost_total
        + dump_cost
        + co2_cost
        + demand_cost
        + extras["Capex_cost_EUR"]
        + extras["Activation_cost_EUR"]
        + extras["Tie_breaker_cost_EUR"]
        + extras["Storage_installation_cost_EUR"]
    )

    objective["OBJ_value_EUR"] = components_sum
    objective["Objective_residual_EUR"] = 0.0

    grid_summary["Energy_from_grid_MWh"] = energy_in
    grid_summary["Energy_to_grid_MWh"] = energy_out
    grid_summary["Net_grid_import_MWh"] = energy_in - energy_out
    grid_summary["Average_purchase_price_EUR_MWh"] = float(energy_cost / energy_in) if energy_in else 0.0
    grid_summary["Average_sell_price_EUR_MWh"] = float(energy_revenue / energy_out) if energy_out else 0.0
    grid_summary["Heat_dumped_MWh"] = heat_dump
    grid_summary["Grid_CO2_emissions_t"] = grid_co2_t
    grid_summary["Total_CO2_emissions_t"] = total_emissions_t
    grid_summary["Heat_demand_MWh"] = heat_demand_mwh

    return orchestrator._flatten_summary(summary_sections)  # type: ignore[attr-defined]


def _apply_design_to_metadata(meta: MutableMapping[str, Any], design: Optional[DesignData]) -> None:
    if not design:
        return

    heat_pumps = meta.get("heat_pumps", [])
    for hp in heat_pumps:
        hp_id = hp.get("id")
        if hp_id is None:
            continue
        design_entry = design.heat_pumps.get(str(hp_id))
        if not design_entry:
            continue
        capacity = float(design_entry.get("capacity_mw", hp.get("max_th", 0.0)))
        hp["max_th"] = capacity
        hp["cap_min"] = capacity
        hp["cap_max"] = capacity
        hp["cap_init"] = capacity
        hp["invest_enabled"] = False

    storage_meta = meta.get("storage")
    if storage_meta and design.storage:
        capacity_mwh = float(design.storage.get("capacity_mwh", storage_meta.get("e_max", 0.0)))
        power_mw = float(design.storage.get("power_mw", storage_meta.get("p_max", 0.0)))
        storage_meta["e_max"] = capacity_mwh
        storage_meta["p_max"] = power_mw
        storage_meta["e_cap_init"] = capacity_mwh
        storage_meta["p_cap_init"] = power_mw
        storage_meta["e_cap_min"] = capacity_mwh
        storage_meta["e_cap_max"] = capacity_mwh
        storage_meta["p_cap_min"] = power_mw
        storage_meta["p_cap_max"] = power_mw
        storage_meta["invest_enabled"] = False



def _register_default_steps() -> None:
    register_workflow_step("PF", _pf_step)
    register_workflow_step("RH", _rh_step)


_register_default_steps()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Simple command line interface for :mod:`energis.run.rolling_horizon`."""

    parser = argparse.ArgumentParser(description="Run PF/RH workflows using merged EnerGIS configs")
    parser.add_argument(
        "configs",
        metavar="CONFIG",
        nargs="+",
        help="Configuration files passed to load_and_merge in the given order",
    )
    parser.add_argument(
        "--print-design",
        action="store_true",
        help="Print extracted design values when available",
    )
    args = parser.parse_args(argv)

    result = run_workflow(args.configs)
    steps = " -> ".join(result.plan.steps)
    print(f"[workflow] Executed steps: {steps}")

    if result.pf_result is not None:
        pf_obj = result.pf_result.costs.get("objective.OBJ_value_EUR") if result.pf_result.costs else None
        print(f"  • PF time steps: {len(result.pf_result.table)}")
        if pf_obj is not None:
            print(f"  • PF objective: {pf_obj}")

    if result.rh_result is not None:
        print(f"  • RH windows: {len(result.rh_result.windows)}")
        print(f"  • RH committed steps: {len(result.rh_result.table)}")

    if args.print_design and result.design is not None:
        hp_parts = ", ".join(sorted(result.design.heat_pumps.keys())) or "none"
        print(f"  • Design heat pumps: {hp_parts}")
        if result.design.storage is not None:
            print(f"  • Storage design: {result.design.storage}")

    return 0


__all__ = [
    "ScenarioResult",
    "WindowResult",
    "RollingHorizonResult",
    "DesignData",
    "WorkflowResult",
    "run_workflow",
    "WorkflowContext",
    "WorkflowPlan",
    "register_workflow_step",
    "unregister_workflow_step",
    "get_registered_workflow_steps",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    raise SystemExit(main())

