from __future__ import annotations

from copy import deepcopy
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
    getattr(model, "HP_INV_Q_wrg")[1].fix(0.0)
    getattr(model, "HP_INV_Q_def")[1].fix(0.0)
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
    assert series["HP_INV_COP"] == pytest.approx([0.0])
    hp_section = summary["heat_pump_HP_INV"]
    assert hp_section["Thermal_capacity_MW"] == pytest.approx(4.0)
    assert hp_section["Build_binary"] == pytest.approx(1.0)
    assert hp_section["Investment_enabled"] is True
    assert hp_section["Average_COP"] == pytest.approx(0.0)
    assert flat["objective.Capex_cost_EUR"] == pytest.approx(capex)
    assert flat["heat_pump_HP_INV.Average_COP"] == pytest.approx(0.0)


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
                    "installation_cost_share": 0.1,
                },
                "terminal": {"policy": "equal", "target_mwh": 2.0},
            },
            "generators": {},
        },
        "scenario": {"horizon": {"enforce": False}},
    }

    model = build_model(table, cfg, dt_h=1.0)
    assert hasattr(model, "TES_terminal")
    target = cfg["system"]["storage"]["soc0_mwh"]
    assert pyo.value(model.TES_terminal.lower) == pytest.approx(target)
    assert pyo.value(model.TES_terminal.upper) == pytest.approx(target)
    assert getattr(model, "TES_terminal_policy") == "equal"

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
    install = (200.0 * 5.0 + 150.0 * 3.0) * 0.1 * (period_frac / 5.0)
    assert pyo.value(model.obj.expr) == pytest.approx(capex + activation + tie + install)

    series, summary, flat = _collect_timeseries_and_summary(table, cfg, 1.0, model)
    objective = summary["objective"]
    assert objective["Capex_cost_EUR"] == pytest.approx(capex)
    assert objective["Activation_cost_EUR"] == pytest.approx(activation)
    assert objective["Tie_breaker_cost_EUR"] == pytest.approx(tie)
    assert objective["Storage_installation_cost_EUR"] == pytest.approx(install)
    assert objective["Objective_residual_EUR"] == pytest.approx(0.0)

    storage_section = summary["storage_TES"]
    assert storage_section["Capacity_MWh"] == pytest.approx(5.0)
    assert storage_section["Power_limit_MW"] == pytest.approx(3.0)
    assert storage_section["Build_binary"] == pytest.approx(1.0)
    assert storage_section["Investment_enabled"] is True
    assert flat["objective.Tie_breaker_cost_EUR"] == pytest.approx(tie)
    assert flat["objective.Storage_installation_cost_EUR"] == pytest.approx(install)


@pytest.mark.skipif(not HAVE_PYOMO, reason="Pyomo not available")
def test_storage_terminal_policy_variants():
    table = _table(1)
    base_cfg = {
        "costs": {"include_co2_cost_in_objective": False, "include_gridcost_in_energy": False, "dump_cost_eur_per_mwh_th": 0.0},
        "grid": {"demand_charge_eur_per_mw_y": 0.0, "energy_fee_eur_mwh": 0.0},
        "storage": {"investment_defaults": {"lifetime_years": 20.0}},
        "system": {
            "heat_pumps": [],
            "storage": {
                "enabled": True,
                "min_energy_mwh": 0.0,
                "max_energy_mwh": 10.0,
                "max_power_mw": 5.0,
                "soc0_mwh": 1.0,
            },
            "generators": {},
        },
    }

    cfg_equal = deepcopy(base_cfg)
    cfg_equal["system"]["storage"]["terminal"] = {"policy": "equal", "target_mwh": 3.0}
    model_eq = build_model(table, cfg_equal, dt_h=1.0)
    assert getattr(model_eq, "TES_terminal_policy") == "equal"
    assert hasattr(model_eq, "TES_terminal")
    assert pyo.value(model_eq.TES_terminal.lower) == pytest.approx(3.0)
    assert pyo.value(model_eq.TES_terminal.upper) == pytest.approx(3.0)

    cfg_geq = deepcopy(base_cfg)
    cfg_geq["system"]["storage"]["terminal"] = {"policy": "geq", "target_mwh": 1.5}
    model_geq = build_model(table, cfg_geq, dt_h=1.0)
    assert getattr(model_geq, "TES_terminal_policy") == "geq"
    assert hasattr(model_geq, "TES_terminal")
    assert model_geq.TES_terminal.lower == pytest.approx(1.5)
    assert model_geq.TES_terminal.upper is None

    cfg_free = deepcopy(base_cfg)
    cfg_free["system"]["storage"]["terminal"] = {"policy": "free"}
    model_free = build_model(table, cfg_free, dt_h=1.0)
    assert getattr(model_free, "TES_terminal_policy") == "free"
    assert not hasattr(model_free, "TES_terminal")
    assert not hasattr(model_free, "TES_terminal_target")


@pytest.mark.skipif(not HAVE_PYOMO, reason="Pyomo not available")
def test_storage_soc_dynamics_with_scaling():
    loss = [0.9, 0.8, 0.0]
    eff_c = [0.95, 0.9, 0.0]
    eff_d = [0.9, 0.85, 0.9]
    active = [1.0, 1.0, 0.0]
    table = _table(
        3,
        {
            "storage_loss_hour": loss,
            "storage_eff_charge": eff_c,
            "storage_eff_discharge": eff_d,
            "storage_capacity_active": active,
        },
    )
    cfg = {
        "costs": {"include_co2_cost_in_objective": False, "include_gridcost_in_energy": False, "dump_cost_eur_per_mwh_th": 0.0},
        "grid": {"demand_charge_eur_per_mw_y": 0.0, "energy_fee_eur_mwh": 0.0},
        "storage": {"investment_defaults": {"lifetime_years": 20.0}},
        "system": {
            "heat_pumps": [],
            "storage": {
                "enabled": True,
                "min_energy_mwh": 0.0,
                "max_energy_mwh": 100.0,
                "max_power_mw": 50.0,
                "soc0_mwh": 1.0,
                "investment": {"enabled": False},
            },
            "generators": {},
        },
    }

    model = build_model(table, cfg, dt_h=1.0)
    cap_e = getattr(model, "TES_cap_energy")
    cap_p = getattr(model, "TES_cap_power")
    build = getattr(model, "TES_build")
    cap_e.fix(20.0)
    cap_p.fix(10.0)
    build.fix(1.0)

    qc = getattr(model, "TES_Qc")
    qd = getattr(model, "TES_Qd")
    soc = getattr(model, "TES_E")
    charge_mode = getattr(model, "TES_charge_mode")
    discharge_mode = getattr(model, "TES_discharge_mode")
    active_var = getattr(model, "TES_active")

    qc[1].fix(2.0)
    qd[1].fix(0.0)
    qc[2].fix(0.0)
    qd[2].fix(1.0)
    qc[3].fix(0.0)
    qd[3].fix(0.0)

    charge_mode[1].fix(1)
    discharge_mode[1].fix(0)
    charge_mode[2].fix(0)
    discharge_mode[2].fix(1)
    charge_mode[3].fix(0)
    discharge_mode[3].fix(0)

    active_var[1].fix(1)
    active_var[2].fix(1)
    active_var[3].fix(0)

    soc_expected = [2.8, 1.0635294118, 0.0]
    soc[1].fix(soc_expected[0])
    soc[2].fix(soc_expected[1])
    soc[3].fix(soc_expected[2])

    for t in model.t:
        cons = getattr(model, "TES_soc")[t]
        assert cons.lower == cons.upper
        assert pyo.value(cons.body - cons.upper) == pytest.approx(0.0, abs=1e-9)

    assert [pyo.value(model.TES_loss[t]) for t in model.t] == pytest.approx(loss)
    assert [pyo.value(model.TES_effc[t]) for t in model.t] == pytest.approx(eff_c)
    assert [pyo.value(model.TES_effd[t]) for t in model.t] == pytest.approx(eff_d)
    assert [pyo.value(model.TES_capacity_active_limit[t]) for t in model.t] == pytest.approx(active)
    assert pyo.value(soc[3]) == pytest.approx(0.0)
