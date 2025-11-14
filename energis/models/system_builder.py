from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd, numpy as np
try:
    import pyomo.environ as pyo
    HAVE_PYOMO = True
except Exception:
    HAVE_PYOMO = False
    pyo = None

from .blocks.heat_pump import HeatPumpBlock
from .blocks.storage   import StorageBlock
from .blocks.thermal_gen import ThermalGeneratorBlock
from .blocks.p2h       import P2HBlock

def _cop_series_from_df(df: pd.DataFrame, wrg_col: str | None, cfg: Dict[str,Any], hp_type: str) -> List[float]:
    copcfg = cfg.get("heat_pumps",{}).get("cop",{})
    dT = float(copcfg.get("deltaT_K", 20.0))
    dTpp = float(copcfg.get("deltaTpp_K", 5.0))
    sink = copcfg.get("sink_defaults", {})
    Ts_out = float(sink.get("Tsink_out_K", 363.15))
    Ts_in  = float(sink.get("Tsink_in_K",  343.15))
    type_par = cfg.get("heat_pumps",{}).get("types",{}).get(hp_type,{})
    eta = float(type_par.get("eta", 0.75))
    FQ  = float(type_par.get("FQ",  0.10))
    if wrg_col and wrg_col in df.columns:
        Tin = pd.to_numeric(df[wrg_col], errors="coerce").ffill().bfill().astype(float)
    else:
        Tin = pd.Series(Ts_in - 10.0, index=df.index)
    Tout = (Tin - dT).clip(lower=1.0)
    # compute COP
    def _lmtd(Th, Tc):
        d1 = max(float(Th) - float(Tc), 1e-6)
        return d1 / max(abs(np.log(max(Th-1e-9, Th)/max(Tc+1e-9, Tc))), 1e-6)
    Ls = _lmtd(Ts_out, Ts_in)
    cop = []
    for i in range(len(df)):
        Tsi = float(Tin.iloc[i]); Tso = float(Tout.iloc[i])
        Lsrc = _lmtd(Tsi, Tso)
        mdts = 0.2*(Ts_out - Tso + 2*dTpp) + 0.2*(Ts_out - Ts_in) + 0.016
        qww  = 0.0014*(Ts_out - Tso + 2*dTpp) - 0.0015*(Ts_out - Ts_in) + 0.039
        A = Ls / max(1e-9, (Ls - Lsrc))
        B = (1 + (mdts + dTpp)/max(1e-9,Ls)) / (1 + (mdts + 0.5*(Tsi - Tso) + 2*dTpp)/max(1e-9,(Ls - Lsrc)))
        val = A * B * eta * (1 - qww) + 1 - eta - FQ
        if (not np.isfinite(val)) or val < 1.01:
            val = 3.0
        cop.append(float(min(max(val,1.01),12.0)))
    return cop

def build_model(df: pd.DataFrame, cfg: Dict[str,Any], dt_h: float = 1.0):
    if not HAVE_PYOMO:
        return None

    T = len(df)
    m = pyo.ConcreteModel(name="EnerGIS_FuelBus")
    m.t = pyo.RangeSet(1, T)

    # --- Params from df
    def s(col): return {i: float(df[col].iloc[i-1]) for i in range(1,T+1)}
    m.price   = pyo.Param(m.t, initialize=s("strompreis_EUR_MWh"), mutable=True)
    m.heatd   = pyo.Param(m.t, initialize=s("waermebedarf_MWth"),  mutable=True)
    m.grid_co2= pyo.Param(m.t, initialize=s("grid_co2_kg_MWh"),    mutable=True)

    # Cost params
    costs = cfg.get("costs", {})
    grid  = cfg.get("grid", {})
    m.energy_fee  = pyo.Param(initialize=float(grid.get("energy_fee_eur_mwh", 0.0)))
    m.co2_price   = pyo.Param(initialize=float(costs.get("co2_price_eur_per_t", 100.0)))
    m.dump_cost   = pyo.Param(initialize=float(costs.get("dump_cost_eur_per_mwh_th", 1.0)))
    m.demand_charge_y = pyo.Param(initialize=float(grid.get("demand_charge_eur_per_mw_y", 0.0)))

    include_gridcost = bool(costs.get("include_gridcost_in_energy", False))
    include_demand   = bool(costs.get("include_demand_charge_in_rh", True))
    include_co2      = bool(costs.get("include_co2_cost_in_objective", True))

    # --- Fuel catalog
    fuels = cfg.get("fuels",{})
    def pfuel(key, default=0.0): return float(fuels.get(key,{}).get("price_eur_mwh", default))
    def efuel(key, default=0.0): return float(fuels.get(key,{}).get("ef_kg_per_mwh_fuel", default))

    # --- Grid vars
    m.P_buy  = pyo.Var(m.t, domain=pyo.NonNegativeReals)
    m.P_sell = pyo.Var(m.t, domain=pyo.NonNegativeReals)
    m.Q_dump = pyo.Var(m.t, domain=pyo.NonNegativeReals)

    # buses: collect flows per medium (FIXED: konsistente Namen!)
    el_in, el_out = [], []
    ht_out, ht_in = [], []
    gas_in, bio_in, waste_in = [], [], []

    syscfg = cfg.get("system", {})

    # Heat pumps
    for hp in syscfg.get("heat_pumps", []):
        if not hp.get("enabled", True): continue
        name = hp.get("id","HP")
        hp_type = hp.get("type","standard")
        wrg_col = hp.get("wrg_source_column")
        COP_series = _cop_series_from_df(df, wrg_col, cfg, hp_type)
        type_par = cfg.get("heat_pumps",{}).get("types",{}).get(hp_type,{})
        min_load = float(type_par.get("min_load", 0.3))
        block = HeatPumpBlock(name, max_th_mw=float(hp.get("max_th_mw", 100.0)), min_load=min_load, COP_series=COP_series)
        fs = block.attach(m, m.t, cfg, {})
        ht_out.append(fs["Q_th_out"]); el_in.append(fs["P_el_in"])

    # Storage (TES)
    sto_cfg = syscfg.get("storage", {"enabled": False})
    if sto_cfg.get("enabled", False):
        block = StorageBlock("TES",
                             e_min=sto_cfg.get("min_energy_mwh", 0.0),
                             e_max=sto_cfg.get("max_energy_mwh", 50000.0),
                             p_max=sto_cfg.get("max_power_mw", 50.0),
                             eff_c=0.95, eff_d=0.95,
                             hourly_loss=0.9999, dt_h=dt_h, soc0=float(sto_cfg.get("soc0_mwh",0.0)))
        fs = block.attach(m, m.t, cfg, {})
        ht_out.append(fs["Q_th_out"]); ht_in.append(fs["Q_th_in"])
        m.TES_SOC = pyo.Reference(fs["SOC"])

    # Generators
    gens = syscfg.get("generators", {})
    fuel_cost_terms, fuel_co2_terms = [], []

    for key, par in gens.items():
        if not par.get("enabled", False): continue
        gpar = cfg.get("generators",{}).get(key,{})
        
        # FIXED: P2H separate behandeln (war doppelt gezählt)
        if key == "p2h":
            block = P2HBlock("P2H", eff=float(gpar.get("el_to_th_eff",0.99)), cap_th_mw=float(par.get("cap_th_mw",10.0)))
            fs = block.attach(m, m.t, cfg, {})
            el_in.append(fs["P_el_in"]); ht_out.append(fs["Q_th_out"])
            continue

        # Normale thermische Generatoren
        block = ThermalGeneratorBlock(key.upper(), th_eff=float(gpar.get("th_eff",0.9)),
                                      el_eff=gpar.get("el_eff", None), cap_th_mw=float(par.get("cap_th_mw",10.0)))
        fs = block.attach(m, m.t, cfg, {})
        ht_out.append(fs["Q_th_out"])
        if fs.get("P_el_out") is not None: el_out.append(fs["P_el_out"])

        fuel_bus = gpar.get("fuel_bus","gas")
        if fuel_bus == "gas":
            gas_in.append(fs["fuel_in"])
            price = pfuel("gas", 0.0); ef = efuel("gas", 0.0)
        elif fuel_bus == "biomass":
            bio_in.append(fs["fuel_in"])
            price = pfuel("biomass", 0.0); ef = efuel("biomass", 0.0)
        elif fuel_bus == "waste":
            waste_in.append(fs["fuel_in"])
            price = pfuel("waste", 0.0); ef = efuel("waste", 0.0)
        else:
            gas_in.append(fs["fuel_in"]); price = 0.0; ef = 0.0
        fuel_cost_terms.append(sum(fs["fuel_in"][t]*price*dt_h for t in m.t))
        fuel_co2_terms.append(sum(fs["fuel_in"][t]*ef*dt_h for t in m.t))

    # FIXED: Sanity Check mit korrektem Variablennamen
    if not ht_out:
        raise RuntimeError(
            "Kein thermischer Erzeuger an den Heat-Bus angeschlossen (ht_out leer). "
            "Bitte System-Config prüfen (enabled/caps) und Attach-Implementierung sicherstellen."
        )

    # Debug-Ausgaben
    print(f"[BUILD] #el_in={len(el_in)}, #el_out={len(el_out)}, #ht_out={len(ht_out)}, #ht_in={len(ht_in)}")

    # --- Bus balances
    m.el_balance = pyo.Constraint(m.t, rule=lambda mm,t:
        mm.P_buy[t] + sum((f[t] for f in el_out), start=0) == sum((f[t] for f in el_in), start=0) + mm.P_sell[t]
    )
    m.ht_balance = pyo.Constraint(m.t, rule=lambda mm,t:
        sum((f[t] for f in ht_out), start=0) + sum((f[t] for f in ht_in), start=0) == mm.heatd[t] + mm.Q_dump[t]
    )
    # fuel buses: consumption must be covered by external supply (here modeled implicitly by cost term only)
    # if desired, you can add explicit supply vars & costs. For now we only ensure non-negativity via component vars.

    # --- Objective
    if include_gridcost:
        buy_price = [m.price[t] + m.energy_fee for t in m.t]
    else:
        buy_price = [m.price[t] for t in m.t]

    energy_cost = sum(dt_h * (m.P_buy[t]*bp - m.P_sell[t]*m.price[t]) for t,bp in zip(m.t, buy_price))
    dump_cost   = m.dump_cost * sum(m.Q_dump[t]*dt_h for t in m.t)
    fuel_costs  = sum(fuel_cost_terms) if fuel_cost_terms else 0
    co2_grid    = sum(m.P_buy[t]*m.grid_co2[t]*dt_h for t in m.t)
    co2_fuel    = sum(fuel_co2_terms) if fuel_co2_terms else 0
    co2_term    = (m.co2_price/1000.0)*(co2_grid+co2_fuel) if include_co2 else 0
    # simple demand charge approximation (peak * annualized factor share by period length fraction)
    period_frac = float(T*dt_h/8760.0)
    m.P_buy_peak = pyo.Var(domain=pyo.NonNegativeReals)
    m.peak_con   = pyo.Constraint(m.t, rule=lambda mm,t: mm.P_buy_peak >= mm.P_buy[t])
    demand_term  = (m.demand_charge_y * period_frac * m.P_buy_peak) if include_demand else 0

    m.obj = pyo.Objective(expr=energy_cost + dump_cost + fuel_costs + co2_term + demand_term, sense=pyo.minimize)
    return m