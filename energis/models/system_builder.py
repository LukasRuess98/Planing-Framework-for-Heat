from __future__ import annotations

from typing import Dict, Any, List
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


def _cop_series_from_table(table: TimeSeriesTable, wrg_col: str | None, cfg: Dict[str, Any], hp_type: str) -> List[float]:
    copcfg = cfg.get("heat_pumps", {}).get("cop", {})
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
            val = 3.0
        cop.append(float(min(max(val, 1.01), 12.0)))
    return cop


def build_model(table: TimeSeriesTable, cfg: Dict[str, Any], dt_h: float = 1.0):
    if not HAVE_PYOMO:
        return None

    T = len(table)
    m = pyo.ConcreteModel(name="EnerGIS_FuelBus")
    m.t = pyo.RangeSet(1, T)

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
        type_par = cfg.get("heat_pumps", {}).get("types", {}).get(hp_type, {})
        min_load = float(type_par.get("min_load", 0.3))
        block = HeatPumpBlock(name, max_th_mw=float(hp.get("max_th_mw", 100.0)), min_load=min_load, COP_series=COP_series)
        fs = block.attach(m, m.t, cfg, {})
        ht_out.append(fs["Q_th_out"])
        el_in.append(fs["P_el_in"])

    sto_cfg = syscfg.get("storage", {"enabled": False})
    if sto_cfg.get("enabled", False):
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
        )
        fs = block.attach(m, m.t, cfg, {})
        ht_out.append(fs["Q_th_out"])
        ht_in.append(fs["Q_th_in"])
        m.TES_SOC = pyo.Reference(fs["SOC"])

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

    period_frac = float(T * dt_h / 8760.0)
    m.P_buy_peak = pyo.Var(domain=pyo.NonNegativeReals)
    m.peak_con = pyo.Constraint(m.t, rule=lambda mm, t: mm.P_buy_peak >= mm.P_buy[t])
    demand_term = (m.demand_charge_y * period_frac * m.P_buy_peak) if include_demand else 0

    m.obj = pyo.Objective(expr=energy_cost + dump_cost + fuel_costs + co2_term + demand_term, sense=pyo.minimize)
    return m

