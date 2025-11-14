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


def _table(hours: int, extra_columns: dict[str, list[float]] | None = None) -> TimeSeriesTable:
    base = datetime(2023, 1, 1, 0)
    idx = [base + timedelta(hours=i) for i in range(hours)]
    columns = ["strompreis_EUR_MWh", "waermebedarf_MWth", "grid_co2_kg_MWh"]
    data = {
        "strompreis_EUR_MWh": [0.0] * hours,
        "waermebedarf_MWth": [0.0] * hours,
        "grid_co2_kg_MWh": [0.0] * hours,
    }
    if extra_columns:
        for key, values in extra_columns.items():
            columns.append(key)
            data[key] = values
    return TimeSeriesTable(idx, columns, data)


@pytest.mark.skipif(not HAVE_PYOMO, reason="Pyomo not available")
def test_heat_pump_investment_costs():
    table = _table(1, {"WRG1_Q_cap": [100.0]})
    cfg = {
        "costs": {"include_co2_cost_in_objective": False, "include_gridcost_in_energy": False, "dump_cost_eur_per_mwh_th": 0.0},
        "grid": {"demand_charge_eur_per_mw_y": 0.0, "energy_fee_eur_mwh": 0.0},
        "heat_pumps": {
            "cop": {"deltaT_K": 20.0, "deltaTpp_K": 5.0, "sink_defaults": {"Tsink_out_K": 350.0, "Tsink_in_K": 340.0}},
            "types": {"standard": {"eta": 0.8, "FQ": 0.1, "min_load": 0.2}},
            "investment_defaults": {
                "capex_eur_per_mw": 0.0,
                "activation_cost_eur": 0.0,
                "lifetime_years": 10.0,
                "tie_breaker_eur_per_mw": 0.0,
            },
        },
        "system": {
            "heat_pumps": [
                {
                    "id": "HP_INV",
                    "enabled": True,
                    "type": "standard",
                    "wrg_capacity_column": "WRG1_Q_cap",
                    "investment": {
                        "enabled": True,
                        "capacity_min_mw": 0.0,
                        "capacity_max_mw": 10.0,
                        "capex_eur_per_mw": 1000.0,
                        "activation_cost_eur": 500.0,
                        "lifetime_years": 10.0,
                        "tie_breaker_eur_per_mw": 1.0,
                    },
                }
            ],
            "storage": {"enabled": False},
            "generators": {},
        },
    }

    model = build_model(table, cfg, dt_h=1.0)
    cap = getattr(model, "HP_INV_cap_mw")
    build = getattr(model, "HP_INV_build")
    cap.fix(4.0)
    build.fix(1.0)
    model.P_buy[1].fix(0.0)
    model.P_sell[1].fix(0.0)
    model.Q_dump[1].fix(0.0)
    model.P_buy_peak.fix(0.0)
    getattr(model, "HP_INV_Q")[1].fix(0.0)
    getattr(model, "HP_INV_Pel")[1].fix(0.0)
    getattr(model, "HP_INV_on")[1].fix(0)

    period_frac = 1.0 / 8760.0
    capex = 1000.0 * 4.0 * (period_frac / 10.0)
    activation = 500.0 * (period_frac / 10.0)
    tie = 1.0 * 4.0
    assert pyo.value(model.obj.expr) == pytest.approx(capex + activation + tie)

    series, summary, flat = _collect_timeseries_and_summary(table, cfg, 1.0, model)
    objective = summary["objective"]
    assert objective["Capex_cost_EUR"] == pytest.approx(capex)
    assert objective["Activation_cost_EUR"] == pytest.approx(activation)
    assert objective["Tie_breaker_cost_EUR"] == pytest.approx(tie)
    assert objective["Objective_residual_EUR"] == pytest.approx(0.0)
    hp_section = summary["heat_pump_HP_INV"]
    assert hp_section["Thermal_capacity_MW"] == pytest.approx(4.0)
    assert hp_section["Build_binary"] == pytest.approx(1.0)
    assert hp_section["Investment_enabled"] is True
    assert flat["objective.Capex_cost_EUR"] == pytest.approx(capex)


@pytest.mark.skipif(not HAVE_PYOMO, reason="Pyomo not available")
def test_storage_terminal_policy_and_costs():
    table = _table(2)
    cfg = {
        "costs": {"include_co2_cost_in_objective": False, "include_gridcost_in_energy": False, "dump_cost_eur_per_mwh_th": 0.0},
        "grid": {"demand_charge_eur_per_mw_y": 0.0, "energy_fee_eur_mwh": 0.0},
        "storage": {
            "investment_defaults": {
                "lifetime_years": 20.0,
                "energy_capex_eur_per_mwh": 0.0,
                "power_capex_eur_per_mw": 0.0,
                "activation_cost_eur": 0.0,
                "tie_breaker_eur_per_mwh": 0.0,
            }
        },
        "system": {
            "heat_pumps": [],
            "storage": {
                "enabled": True,
                "min_energy_mwh": 0.0,
                "max_energy_mwh": 0.0,
                "max_power_mw": 0.0,
                "soc0_mwh": 2.0,
                "investment": {
                    "enabled": True,
                    "energy_capacity_min_mwh": 0.0,
                    "energy_capacity_max_mwh": 10.0,
                    "power_capacity_min_mw": 0.0,
                    "power_capacity_max_mw": 5.0,
                    "energy_capex_eur_per_mwh": 200.0,
                    "power_capex_eur_per_mw": 150.0,
                    "activation_cost_eur": 50.0,
                    "lifetime_years": 5.0,
                    "tie_breaker_eur_per_mwh": 0.5,
                },
            },
            "generators": {},
        },
        "scenario": {"horizon": {"enforce": False}},
    }

    model = build_model(table, cfg, dt_h=1.0)
    assert hasattr(model, "TES_terminal")
    target = cfg["system"]["storage"]["soc0_mwh"]
    assert model.TES_terminal.lower == pytest.approx(target)
    assert model.TES_terminal.upper == pytest.approx(target)

    cap_e = getattr(model, "TES_cap_energy")
    cap_p = getattr(model, "TES_cap_power")
    build = getattr(model, "TES_build")
    cap_e.fix(5.0)
    cap_p.fix(3.0)
    build.fix(1.0)
    for var in (model.P_buy, model.P_sell, model.Q_dump):
        for t in model.t:
            var[t].fix(0.0)
    model.P_buy_peak.fix(0.0)

    period_frac = 2.0 / 8760.0
    capex = (200.0 * 5.0 + 150.0 * 3.0) * (period_frac / 5.0)
    activation = 50.0 * (period_frac / 5.0)
    tie = 0.5 * 5.0
    assert pyo.value(model.obj.expr) == pytest.approx(capex + activation + tie)

    series, summary, flat = _collect_timeseries_and_summary(table, cfg, 1.0, model)
    objective = summary["objective"]
    assert objective["Capex_cost_EUR"] == pytest.approx(capex)
    assert objective["Activation_cost_EUR"] == pytest.approx(activation)
    assert objective["Tie_breaker_cost_EUR"] == pytest.approx(tie)
    assert objective["Objective_residual_EUR"] == pytest.approx(0.0)

    storage_section = summary["storage_TES"]
    assert storage_section["Capacity_MWh"] == pytest.approx(5.0)
    assert storage_section["Power_limit_MW"] == pytest.approx(3.0)
    assert storage_section["Build_binary"] == pytest.approx(1.0)
    assert storage_section["Investment_enabled"] is True
    assert flat["objective.Tie_breaker_cost_EUR"] == pytest.approx(tie)
