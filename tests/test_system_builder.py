from __future__ import annotations

from datetime import datetime, timedelta

import pytest

try:
    import pyomo.environ as pyo
    HAVE_PYOMO = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_PYOMO = False

from energis.models.system_builder import build_model
from energis.run.orchestrator import _collect_timeseries_and_summary
from energis.utils.timeseries import TimeSeriesTable


def _table(hours: int, prices: list[float] | None = None, heat: list[float] | None = None) -> TimeSeriesTable:
    base = datetime(2023, 1, 1)
    idx = [base + timedelta(hours=i) for i in range(hours)]
    cols = ["strompreis_EUR_MWh", "waermebedarf_MWth", "grid_co2_kg_MWh"]
    price_series = prices or [0.0] * hours
    heat_series = heat or [0.0] * hours
    data = {
        "strompreis_EUR_MWh": price_series,
        "waermebedarf_MWth": heat_series,
        "grid_co2_kg_MWh": [0.0] * hours,
    }
    return TimeSeriesTable(idx, cols, data)


@pytest.mark.skipif(not HAVE_PYOMO, reason="Pyomo not available")
def test_grid_mode_gates_mutually_exclusive():
    table = _table(1, prices=[50.0], heat=[0.0])
    cfg = {
        "costs": {
            "include_gridcost_in_energy": False,
            "include_demand_charge_in_rh": True,
            "include_co2_cost_in_objective": False,
            "dump_cost_eur_per_mwh_th": 0.0,
        },
        "grid": {},
        "system": {"heat_pumps": [], "storage": {"enabled": False}, "generators": {}},
    }

    model = build_model(table, cfg, dt_h=1.0)

    assert hasattr(model, "grid_mode")
    assert model.grid_mode[1].domain is pyo.Binary
    assert model.M_GRID.value == pytest.approx(10000.0)
    assert isinstance(model.buy_gate, pyo.Constraint)
    assert isinstance(model.sell_gate, pyo.Constraint)

    model.grid_mode[1].fix(1)
    assert pytest.approx(pyo.value(model.sell_gate[1].upper)) == 0.0
    model.grid_mode[1].fix(0)
    assert pytest.approx(pyo.value(model.buy_gate[1].upper)) == 0.0
    model.grid_mode[1].unfix()


@pytest.mark.skipif(not HAVE_PYOMO, reason="Pyomo not available")
def test_energy_cost_components_and_demand_charge():
    table = _table(2, prices=[50.0, 40.0], heat=[2.0, 0.0])
    cfg = {
        "costs": {
            "include_gridcost_in_energy": True,
            "include_demand_charge_in_rh": True,
            "include_co2_cost_in_objective": False,
            "dump_cost_eur_per_mwh_th": 0.0,
        },
        "grid": {
            "energy_fee_eur_mwh": 5.0,
            "gridcost_eur_mwh": 2.0,
            "sell_floor_eur_mwh": 45.0,
            "sell_haircut_fraction": 0.10,
            "sell_spread_eur_mwh": 5.0,
            "sell_fee_eur_mwh": 1.0,
            "sell_premium_eur_mwh": 2.0,
            "demand_charge_eur_per_mw_y": 100.0,
            "year_fraction": 0.5,
            "big_m_grid_mw": 100.0,
        },
        "fuels": {
            "gas": {"price_eur_mwh": 0.0, "ef_kg_per_mwh_fuel": 0.0},
        },
        "generators": {
            "gen": {"th_eff": 0.0, "el_eff": 1.0, "fuel_bus": "gas"},
            "p2h": {"el_to_th_eff": 1.0},
        },
        "system": {
            "heat_pumps": [],
            "storage": {"enabled": False},
            "generators": {
                "gen": {"enabled": True, "cap_th_mw": 0.0},
                "p2h": {"enabled": True, "cap_th_mw": 10.0},
            },
        },
    }

    model = build_model(table, cfg, dt_h=1.0)

    model.P_buy[1].fix(2.0)
    model.P_sell[1].fix(0.0)
    model.P_buy[2].fix(0.0)
    model.P_sell[2].fix(1.0)
    model.grid_mode[1].fix(1)
    model.grid_mode[2].fix(0)
    model.P_buy_peak.fix(2.0)

    model.Q_dump[1].fix(0.0)
    model.Q_dump[2].fix(0.0)

    gen_Qth = getattr(model, "GEN_Qth")
    gen_fuel = getattr(model, "GEN_fuel")
    gen_Pel = getattr(model, "GEN_Pel")
    gen_Qth[1].fix(0.0)
    gen_Qth[2].fix(0.0)
    gen_fuel[1].fix(0.0)
    gen_fuel[2].fix(1.0)
    gen_Pel[1].fix(0.0)
    gen_Pel[2].fix(1.0)

    p2h_Q = getattr(model, "P2H_Qth")
    p2h_P = getattr(model, "P2H_Pel")
    p2h_Q[1].fix(2.0)
    p2h_P[1].fix(2.0)
    p2h_Q[2].fix(0.0)
    p2h_P[2].fix(0.0)

    expected_buy_price = 50.0 + 5.0 + 2.0
    expected_sell_price = max(40.0 - 5.0, 45.0)
    expected_sell_price = expected_sell_price * (1.0 - 0.10)
    expected_sell_price = expected_sell_price - 1.0 + 2.0
    expected_sell_price = max(expected_sell_price, 0.0)

    expected_energy_cost = 2.0 * expected_buy_price
    expected_energy_revenue = 1.0 * expected_sell_price
    expected_demand_charge = 100.0 * 0.5 * 2.0

    expected_objective = expected_energy_cost - expected_energy_revenue + expected_demand_charge

    assert pyo.value(model.obj.expr) == pytest.approx(expected_objective)

    series, summary, flat = _collect_timeseries_and_summary(table, cfg, 1.0, model)
    objective = summary["objective"]
    grid_section = summary["grid"]

    assert objective["Grid_energy_cost_EUR"] == pytest.approx(expected_energy_cost)
    assert objective["Grid_sell_revenue_EUR"] == pytest.approx(expected_energy_revenue)
    assert objective["Grid_net_cost_EUR"] == pytest.approx(expected_energy_cost - expected_energy_revenue)
    assert objective["Demand_charge_cost_EUR"] == pytest.approx(expected_demand_charge)
    assert objective["Demand_charge_year_fraction"] == pytest.approx(0.5)
    assert objective["Objective_residual_EUR"] == pytest.approx(0.0)

    assert grid_section["Average_purchase_price_EUR_MWh"] == pytest.approx(expected_buy_price)
    assert grid_section["Average_sell_price_EUR_MWh"] == pytest.approx(expected_sell_price)

    assert flat["objective.Grid_energy_cost_EUR"] == pytest.approx(expected_energy_cost)
    assert series["P_buy_MW"] == pytest.approx([2.0, 0.0])
    assert series["P_sell_MW"] == pytest.approx([0.0, 1.0])
