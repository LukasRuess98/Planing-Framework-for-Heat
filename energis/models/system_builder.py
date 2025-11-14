from __future__ import annotations

from typing import Dict, Any, List, Optional, Sequence
import math

try:
    import pyomo.environ as pyo
    HAVE_PYOMO = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_PYOMO = False
    pyo = None

from energis.utils.timeseries import TimeSeriesTable
from .blocks.heat_pump import HeatPumpBlock
from .blocks.storage import StorageBlock
from .blocks.thermal_gen import ThermalGeneratorBlock
from .blocks.p2h import P2HBlock


def _cop_series_from_table(
    table: TimeSeriesTable, wrg_col: str | None, cfg: Dict[str, Any], hp_type: str
) -> List[float]:
    copcfg = cfg.get("heat_pumps", {}).get("cop", {})
    tables_cfg = copcfg.get("tables", {})
    table_spec = tables_cfg.get(hp_type) or tables_cfg.get("default")

    def _validate_axis(values: Sequence[float], axis_name: str) -> List[float]:
        axis = [float(v) for v in values]
        if not axis:
            raise ValueError(f"COP table axis '{axis_name}' is empty")
        if any(not math.isfinite(v) for v in axis):
            raise ValueError(f"COP table axis '{axis_name}' enthält ungültige Werte")
        if axis != sorted(axis):
            raise ValueError(f"COP table axis '{axis_name}' muss aufsteigend sortiert sein")
        return axis

    def _locate_interval(points: List[float], value: float, clamp: bool, axis_name: str) -> tuple[int, int, float]:
        if len(points) == 1:
            return 0, 0, 0.0
        if value <= points[0]:
            if clamp:
                return 0, 0, 0.0
            raise ValueError(f"Wert {value} liegt unterhalb der COP-Tabelle für {axis_name}")
        if value >= points[-1]:
            if clamp:
                idx = len(points) - 1
                return idx, idx, 0.0
            raise ValueError(f"Wert {value} liegt oberhalb der COP-Tabelle für {axis_name}")
        for i in range(len(points) - 1):
            lo = points[i]
            hi = points[i + 1]
            if lo <= value <= hi or math.isclose(value, lo) or math.isclose(value, hi):
                span = max(hi - lo, 1e-12)
                frac = (value - lo) / span
                return i, i + 1, min(max(frac, 0.0), 1.0)
        # Should not happen due to bounds above
        raise ValueError(f"Wert {value} konnte nicht in COP-Achse {axis_name} zugeordnet werden")

    def _interp1d(points: List[float], values: List[float], i0: int, i1: int, frac: float) -> float:
        if i0 == i1:
            return values[i0]
        v0 = values[i0]
        v1 = values[i1]
        return v0 + frac * (v1 - v0)

    def _interp2d(
        x_points: List[float],
        matrix: List[List[float]],
        x_idx0: int,
        x_idx1: int,
        x_frac: float,
        y_idx0: int,
        y_idx1: int,
        y_frac: float,
    ) -> float:
        if y_idx0 == y_idx1:
            return _interp1d(x_points, matrix[y_idx0], x_idx0, x_idx1, x_frac)
        # Interpolate along x for both y positions, then between them
        row0 = matrix[y_idx0]
        row1 = matrix[y_idx1]
        v0 = _interp1d(x_points, row0, x_idx0, x_idx1, x_frac)
        v1 = _interp1d(x_points, row1, x_idx0, x_idx1, x_frac)
        return v0 + y_frac * (v1 - v0)

    def _series_from_column(column: str | None, default: float | None) -> List[float]:
        if column and column in table.columns:
            return [float(table[column][i]) for i in range(len(table))]
        if default is not None:
            return [float(default) for _ in range(len(table))]
        raise KeyError(f"Benötigte Spalte {column!r} für COP-Berechnung fehlt")

    if table_spec:
        x_points = _validate_axis(table_spec.get("x") or table_spec.get("source_temperatures_K", []), "x")
        y_points_raw: Sequence[float] | None = table_spec.get("y") or table_spec.get("sink_temperatures_K")
        has_y = bool(y_points_raw)
        y_points = _validate_axis(y_points_raw, "y") if has_y else None
        values_raw = table_spec.get("values")
        if values_raw is None:
            raise ValueError("COP-Tabelle benötigt einen 'values'-Eintrag")
        if has_y:
            matrix = [[float(v) for v in row] for row in values_raw]
            if len(matrix) != len(y_points):
                raise ValueError("COP-Tabelle: Anzahl der Zeilen passt nicht zur y-Achse")
            for row in matrix:
                if len(row) != len(x_points):
                    raise ValueError("COP-Tabelle: Jede Zeile muss gleich viele Werte wie die x-Achse besitzen")
        else:
            vector = [float(v) for v in values_raw]
            if len(vector) != len(x_points):
                raise ValueError("COP-Tabelle (1D): Anzahl der Werte passt nicht zur x-Achse")
            matrix = [vector]
            y_points = [0.0]
            has_y = False

        clamp_default = bool(table_spec.get("clamp", True))
        clamp_x = bool(table_spec.get("clamp_x", clamp_default))
        clamp_y = bool(table_spec.get("clamp_y", clamp_default))

        sink_defaults = copcfg.get("sink_defaults", {})
        Ts_out = float(sink_defaults.get("Tsink_out_K", 363.15))

        x_series = _series_from_column(table_spec.get("x_column") or wrg_col, table_spec.get("x_default"))
        y_column = table_spec.get("y_column")
        y_series = _series_from_column(y_column, table_spec.get("y_default", Ts_out)) if has_y else [0.0] * len(table)

        cop_min = float(table_spec.get("cop_min", copcfg.get("cop_min", 1.01)))
        cop_max = float(table_spec.get("cop_max", copcfg.get("cop_max", 12.0)))

        result: List[float] = []
        for xv, yv in zip(x_series, y_series):
            x_idx0, x_idx1, x_frac = _locate_interval(x_points, float(xv), clamp_x, "x")
            if has_y:
                y_idx0, y_idx1, y_frac = _locate_interval(y_points, float(yv), clamp_y, "y")
            else:
                y_idx0 = y_idx1 = 0
                y_frac = 0.0
            val = _interp2d(
                x_points,
                matrix,
                x_idx0,
                x_idx1,
                x_frac,
                y_idx0,
                y_idx1,
                y_frac,
            )
            if not math.isfinite(val):
                raise ValueError("COP-Tabelle lieferte keinen gültigen Wert")
            result.append(float(min(max(val, cop_min), cop_max)))
        return result

    # Fallback auf analytische Berechnung
    dT = float(copcfg.get("deltaT_K", 20.0))
    dTpp = float(copcfg.get("deltaTpp_K", 5.0))
    sink = copcfg.get("sink_defaults", {})
    Ts_out = float(sink.get("Tsink_out_K", 363.15))
    Ts_in = float(sink.get("Tsink_in_K", 343.15))
    type_par = cfg.get("heat_pumps", {}).get("types", {}).get(hp_type, {})
    eta = float(type_par.get("eta", 0.75))
    FQ = float(type_par.get("FQ", 0.10))

    if wrg_col and wrg_col in table.columns:
        temps = table[wrg_col]
    else:
        temps = [Ts_in - 10.0 for _ in range(len(table))]

    Tout = [max(t - dT, 1.0) for t in temps]

    def _lmtd(Th: float, Tc: float) -> float:
        d1 = max(Th - Tc, 1e-6)
        numerator = d1
        denominator = abs(math.log(max(Th - 1e-9, Th) / max(Tc + 1e-9, Tc)))
        return numerator / max(denominator, 1e-6)

    Ls = _lmtd(Ts_out, Ts_in)
    cop: List[float] = []
    for Tin, Tout_i in zip(temps, Tout):
        Lsrc = _lmtd(Tin, Tout_i)
        mdts = 0.2 * (Ts_out - Tout_i + 2 * dTpp) + 0.2 * (Ts_out - Ts_in) + 0.016
        qww = 0.0014 * (Ts_out - Tout_i + 2 * dTpp) - 0.0015 * (Ts_out - Ts_in) + 0.039
        A = Ls / max(1e-9, Ls - Lsrc)
        B = (1 + (mdts + dTpp) / max(1e-9, Ls)) / (1 + (mdts + 0.5 * (Tin - Tout_i) + 2 * dTpp) / max(1e-9, (Ls - Lsrc)))
        val = A * B * eta * (1 - qww) + 1 - eta - FQ
        if not math.isfinite(val) or val < 1.01:
            val = float(copcfg.get("cop_fallback", 3.0))
        cop.append(float(min(max(val, 1.01), 12.0)))
    return cop


def build_model(table: TimeSeriesTable, cfg: Dict[str, Any], dt_h: float = 1.0):
    if not HAVE_PYOMO:
        return None

    T = len(table)
    m = pyo.ConcreteModel(name="EnerGIS_FuelBus")
    m.t = pyo.RangeSet(1, T)
    period_frac = float(T * dt_h / 8760.0)

    def series_dict(name: str) -> Dict[int, float]:
        values = table[name]
        return {i + 1: float(values[i]) for i in range(T)}

    m.price = pyo.Param(m.t, initialize=series_dict("strompreis_EUR_MWh"), mutable=True)
    m.heatd = pyo.Param(m.t, initialize=series_dict("waermebedarf_MWth"), mutable=True)
    m.grid_co2 = pyo.Param(m.t, initialize=series_dict("grid_co2_kg_MWh"), mutable=True)

    costs = cfg.get("costs", {})
    grid = cfg.get("grid", {})
    m.energy_fee = pyo.Param(initialize=float(grid.get("energy_fee_eur_mwh", 0.0)))
    m.co2_price = pyo.Param(initialize=float(costs.get("co2_price_eur_per_t", 100.0)))
    m.dump_cost = pyo.Param(initialize=float(costs.get("dump_cost_eur_per_mwh_th", 1.0)))
    m.demand_charge_y = pyo.Param(initialize=float(grid.get("demand_charge_eur_per_mw_y", 0.0)))

    include_gridcost = bool(costs.get("include_gridcost_in_energy", False))
    include_demand = bool(costs.get("include_demand_charge_in_rh", True))
    include_co2 = bool(costs.get("include_co2_cost_in_objective", True))

    fuels = cfg.get("fuels", {})

    def pfuel(key: str, default: float = 0.0) -> float:
        return float(fuels.get(key, {}).get("price_eur_mwh", default))

    def efuel(key: str, default: float = 0.0) -> float:
        return float(fuels.get(key, {}).get("ef_kg_per_mwh_fuel", default))

    m.P_buy = pyo.Var(m.t, domain=pyo.NonNegativeReals)
    m.P_sell = pyo.Var(m.t, domain=pyo.NonNegativeReals)
    m.Q_dump = pyo.Var(m.t, domain=pyo.NonNegativeReals)

    el_in: List = []
    el_out: List = []
    ht_out: List = []
    ht_in: List = []
    gas_in: List = []
    bio_in: List = []
    waste_in: List = []

    syscfg = cfg.get("system", {})

    hp_defaults = cfg.get("heat_pumps", {})
    hp_inv_defaults = hp_defaults.get("investment_defaults", {})

    capex_terms: List = []
    activation_terms: List = []
    tie_breaker_terms: List = []

    for hp in syscfg.get("heat_pumps", []):
        if not hp.get("enabled", True):
            continue
        name = hp.get("id", "HP")
        hp_type = hp.get("type", "standard")
        wrg_col = None
        if hp.get("wrg_source_column"):
            wrg_col = hp.get("wrg_source_column")
            if wrg_col not in table.columns and f"{wrg_col}_K" in table.columns:
                wrg_col = f"{wrg_col}_K"
        COP_series = _cop_series_from_table(table, wrg_col, cfg, hp_type)
        wrg_cap_col: Optional[str] = hp.get("wrg_capacity_column")
        if wrg_cap_col is None and hp.get("wrg_source_column"):
            prefix = str(hp.get("wrg_source_column")).split("_T")[0]
            candidate = f"{prefix}_Q_cap"
            if candidate in table.columns:
                wrg_cap_col = candidate
        wrg_caps = None
        if wrg_cap_col and wrg_cap_col in table.columns:
            wrg_caps = {i + 1: float(table[wrg_cap_col][i]) for i in range(T)}

        inv_cfg = dict(hp_inv_defaults)
        inv_cfg.update(hp.get("investment", {}))
        invest_enabled = bool(inv_cfg.get("enabled", False))
        cap_min = float(inv_cfg.get("capacity_min_mw", hp.get("min_th_mw", 0.0)))
        cap_max = float(inv_cfg.get("capacity_max_mw", hp.get("max_th_mw", 0.0)))
        existing_cap = float(hp.get("max_th_mw", cap_max))
        cap_init = float(
            inv_cfg.get(
                "initial_capacity_mw",
                existing_cap if not invest_enabled else max(cap_min, min(existing_cap, cap_max)),
            )
        )
        type_cfg = cfg.get("heat_pumps", {}).get("types", {})
        type_par = type_cfg.get(hp_type, {})
        min_load = float(type_par.get("min_load", 0.3))
        cop_default = float(type_par.get("COPdefault", cfg.get("heat_pumps", {}).get("cop", {}).get("cop_fallback", 3.0)))
        if not math.isfinite(cop_default) or cop_default <= 0:
            cop_default = 3.0

        block = HeatPumpBlock(
            name,
            min_load=min_load,
            cop_series=COP_series,
            capacity_min_mw=cap_min,
            capacity_max_mw=cap_max,
            capacity_init_mw=cap_init,
            investable=invest_enabled,
            wrg_cap_series=wrg_caps,
            cop_default=cop_default,
        )
        fs = block.attach(m, m.t, cfg, {})
        ht_out.append(fs["Q_th_out"])
        el_in.append(fs["P_el_in"])

        cap_var = fs.get("capacity")
        build_var = fs.get("build")
        if cap_var is not None and build_var is not None:
            lifetime = float(inv_cfg.get("lifetime_years", hp_inv_defaults.get("lifetime_years", 20.0)))
            capex = float(inv_cfg.get("capex_eur_per_mw", hp_inv_defaults.get("capex_eur_per_mw", 0.0)))
            activation = float(inv_cfg.get("activation_cost_eur", hp_inv_defaults.get("activation_cost_eur", 0.0)))
            tie_breaker = float(inv_cfg.get("tie_breaker_eur_per_mw", hp_inv_defaults.get("tie_breaker_eur_per_mw", 0.0)))
            if lifetime > 0:
                annual_factor = period_frac / lifetime
            else:
                annual_factor = 0.0
            capex_terms.append(cap_var * capex * annual_factor)
            activation_terms.append(build_var * activation * annual_factor)
            if tie_breaker:
                tie_breaker_terms.append(cap_var * tie_breaker)

    sto_cfg = syscfg.get("storage", {"enabled": False})
    if sto_cfg.get("enabled", False):
        sto_defaults = cfg.get("storage", {}).get("investment_defaults", {})
        sto_inv = dict(sto_defaults)
        sto_inv.update(sto_cfg.get("investment", {}))
        invest_enabled = bool(sto_inv.get("enabled", False))
        e_cap_min = float(sto_inv.get("energy_capacity_min_mwh", sto_cfg.get("min_energy_mwh", 0.0)))
        e_cap_max = float(sto_inv.get("energy_capacity_max_mwh", sto_cfg.get("max_energy_mwh", 50000.0)))
        p_cap_min = float(sto_inv.get("power_capacity_min_mw", sto_cfg.get("min_power_mw", 0.0)))
        p_cap_max = float(sto_inv.get("power_capacity_max_mw", sto_cfg.get("max_power_mw", 50.0)))
        e_cap_init = float(
            sto_inv.get(
                "initial_energy_capacity_mwh",
                sto_cfg.get("max_energy_mwh", e_cap_max) if not invest_enabled else max(e_cap_min, min(e_cap_max, sto_cfg.get("max_energy_mwh", e_cap_max))),
            )
        )
        p_cap_init = float(
            sto_inv.get(
                "initial_power_capacity_mw",
                sto_cfg.get("max_power_mw", p_cap_max) if not invest_enabled else max(p_cap_min, min(p_cap_max, sto_cfg.get("max_power_mw", p_cap_max))),
            )
        )
        terminal_target = sto_cfg.get("terminal_soc_mwh")
        if terminal_target is None:
            horizon_cfg = cfg.get("scenario", {}).get("horizon", {})
            if not bool(horizon_cfg.get("enforce", True)):
                terminal_target = float(sto_cfg.get("soc0_mwh", 0.0))
        block = StorageBlock(
            "TES",
            e_min=sto_cfg.get("min_energy_mwh", 0.0),
            e_max=sto_cfg.get("max_energy_mwh", 50000.0),
            p_max=sto_cfg.get("max_power_mw", 50.0),
            eff_c=0.95,
            eff_d=0.95,
            hourly_loss=0.9999,
            dt_h=dt_h,
            soc0=float(sto_cfg.get("soc0_mwh", 0.0)),
            investable=invest_enabled,
            e_cap_min=e_cap_min,
            e_cap_max=e_cap_max,
            p_cap_min=p_cap_min,
            p_cap_max=p_cap_max,
            e_cap_init=e_cap_init,
            p_cap_init=p_cap_init,
            terminal_target=terminal_target,
        )
        fs = block.attach(m, m.t, cfg, {})
        ht_out.append(fs["Q_th_out"])
        ht_in.append(fs["Q_th_in"])
        m.TES_SOC = pyo.Reference(fs["SOC"])
        cap_var = fs.get("cap_energy")
        pow_var = fs.get("cap_power")
        build_var = fs.get("build")
        lifetime = float(sto_inv.get("lifetime_years", sto_defaults.get("lifetime_years", 20.0)))
        e_capex = float(sto_inv.get("energy_capex_eur_per_mwh", sto_defaults.get("energy_capex_eur_per_mwh", 0.0)))
        p_capex = float(sto_inv.get("power_capex_eur_per_mw", sto_defaults.get("power_capex_eur_per_mw", 0.0)))
        activation = float(sto_inv.get("activation_cost_eur", sto_defaults.get("activation_cost_eur", 0.0)))
        tie_breaker = float(sto_inv.get("tie_breaker_eur_per_mwh", sto_defaults.get("tie_breaker_eur_per_mwh", 0.0)))
        if lifetime > 0:
            annual_factor = period_frac / lifetime
        else:
            annual_factor = 0.0
        if cap_var is not None:
            capex_terms.append(cap_var * e_capex * annual_factor)
            if tie_breaker:
                tie_breaker_terms.append(cap_var * tie_breaker)
        if pow_var is not None:
            capex_terms.append(pow_var * p_capex * annual_factor)
        if build_var is not None:
            activation_terms.append(build_var * activation * annual_factor)

    gens = syscfg.get("generators", {})
    fuel_cost_terms: List = []
    fuel_co2_terms: List = []

    for key, par in gens.items():
        if not par.get("enabled", False):
            continue
        gpar = cfg.get("generators", {}).get(key, {})
        if key == "p2h":
            block = P2HBlock("P2H", eff=float(gpar.get("el_to_th_eff", 0.99)), cap_th_mw=float(par.get("cap_th_mw", 10.0)))
            fs = block.attach(m, m.t, cfg, {})
            el_in.append(fs["P_el_in"])
            ht_out.append(fs["Q_th_out"])
            continue

        block = ThermalGeneratorBlock(
            key.upper(), th_eff=float(gpar.get("th_eff", 0.9)), el_eff=gpar.get("el_eff", None), cap_th_mw=float(par.get("cap_th_mw", 10.0))
        )
        fs = block.attach(m, m.t, cfg, {})
        ht_out.append(fs["Q_th_out"])
        if fs.get("P_el_out") is not None:
            el_out.append(fs["P_el_out"])

        fuel_bus = gpar.get("fuel_bus", "gas")
        if fuel_bus == "gas":
            bus_list = gas_in
            price = pfuel("gas", 0.0)
            ef = efuel("gas", 0.0)
        elif fuel_bus == "biomass":
            bus_list = bio_in
            price = pfuel("biomass", 0.0)
            ef = efuel("biomass", 0.0)
        elif fuel_bus == "waste":
            bus_list = waste_in
            price = pfuel("waste", 0.0)
            ef = efuel("waste", 0.0)
        else:
            bus_list = gas_in
            price = 0.0
            ef = 0.0
        bus_list.append(fs["fuel_in"])
        fuel_cost_terms.append(sum(fs["fuel_in"][t] * price * dt_h for t in m.t))
        fuel_co2_terms.append(sum(fs["fuel_in"][t] * ef * dt_h for t in m.t))

    if not ht_out:
        raise RuntimeError(
            "Kein thermischer Erzeuger an den Heat-Bus angeschlossen (ht_out leer). Bitte System-Config prüfen."
        )

    print(f"[BUILD] #el_in={len(el_in)}, #el_out={len(el_out)}, #ht_out={len(ht_out)}, #ht_in={len(ht_in)}")

    m.el_balance = pyo.Constraint(
        m.t,
        rule=lambda mm, t: mm.P_buy[t] + sum((f[t] for f in el_out), start=0) == sum((f[t] for f in el_in), start=0) + mm.P_sell[t],
    )
    m.ht_balance = pyo.Constraint(
        m.t,
        rule=lambda mm, t: sum((f[t] for f in ht_out), start=0) + sum((f[t] for f in ht_in), start=0) == mm.heatd[t] + mm.Q_dump[t],
    )

    if include_gridcost:
        buy_price = [table["strompreis_EUR_MWh"][i] + float(m.energy_fee.value) for i in range(T)]
    else:
        buy_price = table["strompreis_EUR_MWh"]

    energy_cost = sum(dt_h * (m.P_buy[t] * bp - m.P_sell[t] * table["strompreis_EUR_MWh"][t - 1]) for t, bp in zip(m.t, buy_price))
    dump_cost = m.dump_cost * sum(m.Q_dump[t] * dt_h for t in m.t)
    fuel_costs = sum(fuel_cost_terms) if fuel_cost_terms else 0
    co2_grid = sum(m.P_buy[t] * table["grid_co2_kg_MWh"][t - 1] * dt_h for t in m.t)
    co2_fuel = sum(fuel_co2_terms) if fuel_co2_terms else 0
    co2_term = (m.co2_price / 1000.0) * (co2_grid + co2_fuel) if include_co2 else 0

    m.P_buy_peak = pyo.Var(domain=pyo.NonNegativeReals)
    m.peak_con = pyo.Constraint(m.t, rule=lambda mm, t: mm.P_buy_peak >= mm.P_buy[t])
    demand_term = (m.demand_charge_y * period_frac * m.P_buy_peak) if include_demand else 0

    capex_total = sum(capex_terms) if capex_terms else 0
    activation_total = sum(activation_terms) if activation_terms else 0
    tie_break_total = sum(tie_breaker_terms) if tie_breaker_terms else 0

    m.capex_cost_expr = capex_total
    m.activation_cost_expr = activation_total
    m.tie_break_cost_expr = tie_break_total

    m.obj = pyo.Objective(
        expr=energy_cost + dump_cost + fuel_costs + co2_term + demand_term + capex_total + activation_total + tie_break_total,
        sense=pyo.minimize,
    )
    return m

