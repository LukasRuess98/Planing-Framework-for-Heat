from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timedelta
import copy
from typing import List

import pytest

from energis.run import rolling_horizon as rh
from energis.utils.timeseries import TimeSeriesTable


def _make_table(n_steps: int) -> TimeSeriesTable:
    index = [datetime(2023, 1, 1) + timedelta(hours=i) for i in range(n_steps)]
    data = {
        "waermebedarf_MWth": [float(10 + i) for i in range(n_steps)],
    }
    return TimeSeriesTable(index, list(data.keys()), data)


@pytest.fixture
def simple_config() -> dict:
    return {
        "run": {"dt_h": 1.0, "solver": "dummy"},
        "site": {"input_xlsx": "unused.xlsx"},
        "system": {
            "heat_pumps": [
                {"id": "HP1", "max_th_mw": 100.0, "min_th_mw": 10.0, "investment": {}},
            ],
            "storage": {
                "enabled": True,
                "max_energy_mwh": 100.0,
                "max_power_mw": 30.0,
                "soc0_mwh": 0.0,
                "investment": {"enabled": True},
            },
        },
    }


def test_pf_only_workflow(monkeypatch: pytest.MonkeyPatch, simple_config: dict) -> None:
    config = copy.deepcopy(simple_config)
    config["scenario"] = {"run_mode": "PF_ONLY"}

    table = _make_table(4)

    def fake_loader(path: str, site_cfg: dict, dt_hours: float) -> TimeSeriesTable:
        assert dt_hours == 1.0
        return table

    def fake_solve(table_arg, cfg, dt_h, solver_name):
        assert solver_name == "dummy"
        assert dt_h == 1.0
        assert table_arg.index == table.index
        series = OrderedDict({"TES_SOC_MWh": [0.0] * len(table_arg)})
        summary = OrderedDict(
            {
                "objective": OrderedDict(),
                "heat_pump_HP1": OrderedDict(
                    [
                        ("Thermal_capacity_MW", 5.0),
                        ("Build_binary", 1.0),
                    ]
                ),
                "storage_TES": OrderedDict(
                    [
                        ("Capacity_MWh", 20.0),
                        ("Power_limit_MW", 5.0),
                        ("Build_binary", 1.0),
                    ]
                ),
            }
        )
        costs = {"objective.OBJ_value_EUR": 0.0}
        solver = {"status": "ok"}
        return rh.ScenarioResult(table_arg, series, summary, costs, solver)

    monkeypatch.setattr(rh, "load_input_excel", fake_loader)
    monkeypatch.setattr(rh, "_solve_scenario", fake_solve)

    result = rh.run_workflow([], overrides=config)

    assert result.pf_result is not None
    assert result.rh_result is None
    assert result.design is not None
    assert result.plan.steps == ["PF"]
    assert result.design.heat_pumps["HP1"]["capacity_mw"] == pytest.approx(5.0)
    assert result.design.storage["capacity_mwh"] == pytest.approx(20.0)


def test_rh_only_workflow_aggregates(monkeypatch: pytest.MonkeyPatch, simple_config: dict) -> None:
    config = copy.deepcopy(simple_config)
    config["scenario"] = {
        "workflow": ["RH"],
        "rolling_horizon": {"heat_horizon_hours": 4.0, "step_hours": 2.0, "terminal_policy": "free"},
    }

    table = _make_table(5)

    monkeypatch.setattr(rh, "load_input_excel", lambda *args, **kwargs: table)

    window_series = [
        OrderedDict({"TES_SOC_MWh": [0.0, 1.0, 2.0, 3.0], "P_buy_MW": [1.0, 1.0, 1.0, 1.0]}),
        OrderedDict({"TES_SOC_MWh": [1.0, 2.0, 3.0, 4.0], "P_buy_MW": [2.0, 2.0, 2.0, 2.0]}),
        OrderedDict({"TES_SOC_MWh": [2.0], "P_buy_MW": [3.0]}),
    ]

    expected_soc = [0.0, 1.0, 2.0]

    call_state = {"idx": 0}

    def fake_solve(table_arg, cfg, dt_h, solver_name):
        idx = call_state["idx"]
        call_state["idx"] += 1
        assert cfg["inputs"]["SOC_init"] == pytest.approx(expected_soc[idx])
        series = window_series[idx]
        summary = OrderedDict({"objective": OrderedDict()})
        costs = {"objective.OBJ_value_EUR": float(idx)}
        solver = {"status": "ok"}
        return rh.ScenarioResult(table_arg, series, summary, costs, solver)

    monkeypatch.setattr(rh, "_solve_scenario", fake_solve)

    result = rh.run_workflow([], overrides=config)

    assert result.pf_result is None
    assert result.rh_result is not None
    assert result.plan.steps == ["RH"]
    assert len(result.rh_result.windows) == 3
    assert result.rh_result.series["TES_SOC_MWh"] == [0.0, 1.0, 1.0, 2.0, 2.0]
    assert result.rh_result.series["P_buy_MW"] == [1.0, 1.0, 2.0, 2.0, 3.0]


def test_pf_then_rh_fix_design(monkeypatch: pytest.MonkeyPatch, simple_config: dict) -> None:
    config = copy.deepcopy(simple_config)
    config["scenario"] = {
        "run_mode": "PF_THEN_RH",
        "rolling_horizon": {"HEAT_HORIZON_HOURS": 4.0, "STEP_HOURS": 2.0, "terminal_policy": "free"},
    }

    table = _make_table(5)
    monkeypatch.setattr(rh, "load_input_excel", lambda *args, **kwargs: table)

    pf_summary = OrderedDict(
        {
            "objective": OrderedDict(),
            "heat_pump_HP1": OrderedDict(
                [
                    ("Thermal_capacity_MW", 5.0),
                    ("Build_binary", 1.0),
                ]
            ),
            "storage_TES": OrderedDict(
                [
                    ("Capacity_MWh", 20.0),
                    ("Power_limit_MW", 5.0),
                    ("Build_binary", 1.0),
                ]
            ),
        }
    )

    window_series = [
        OrderedDict({"TES_SOC_MWh": [0.0, 1.0, 2.0, 3.0], "P_buy_MW": [1.0, 1.0, 1.0, 1.0]}),
        OrderedDict({"TES_SOC_MWh": [1.0, 2.0, 3.0, 4.0], "P_buy_MW": [2.0, 2.0, 2.0, 2.0]}),
        OrderedDict({"TES_SOC_MWh": [2.0], "P_buy_MW": [3.0]}),
    ]

    expected_soc = [0.0, 1.0, 2.0]
    call_state = {"idx": 0}

    def fake_solve(table_arg, cfg, dt_h, solver_name):
        idx = call_state["idx"]
        call_state["idx"] += 1
        if idx == 0:
            series = OrderedDict({"TES_SOC_MWh": [0.0] * len(table_arg)})
            costs = {"objective.OBJ_value_EUR": 0.0}
            return rh.ScenarioResult(table_arg, series, pf_summary, costs, {"status": "ok"})

        window_idx = idx - 1
        assert cfg["inputs"]["SOC_init"] == pytest.approx(expected_soc[window_idx])
        hp_cfg = cfg["system"]["heat_pumps"][0]
        assert hp_cfg["investment"]["enabled"] is False
        assert hp_cfg["max_th_mw"] == pytest.approx(5.0)
        assert hp_cfg["min_th_mw"] == pytest.approx(5.0)
        assert hp_cfg["investment"]["capacity_min_mw"] == pytest.approx(5.0)
        assert hp_cfg["investment"]["capacity_max_mw"] == pytest.approx(5.0)
        storage_cfg = cfg["system"]["storage"]
        assert storage_cfg["investment"]["enabled"] is False
        assert storage_cfg["investment"]["energy_capacity_min_mwh"] == pytest.approx(20.0)
        assert storage_cfg["investment"]["energy_capacity_max_mwh"] == pytest.approx(20.0)
        assert storage_cfg["terminal"]["policy"] == "free"
        series = window_series[window_idx]
        costs = {"objective.OBJ_value_EUR": float(window_idx)}
        summary = OrderedDict({"objective": OrderedDict()})
        return rh.ScenarioResult(table_arg, series, summary, costs, {"status": "ok"})

    monkeypatch.setattr(rh, "_solve_scenario", fake_solve)

    result = rh.run_workflow([], overrides=config)

    assert result.pf_result is not None
    assert result.design is not None
    assert result.rh_result is not None
    assert result.plan.steps == ["PF", "RH"]
    assert result.rh_result.series["TES_SOC_MWh"] == [0.0, 1.0, 1.0, 2.0, 2.0]
    assert result.design.heat_pumps["HP1"]["capacity_mw"] == pytest.approx(5.0)
    assert result.design.storage["power_mw"] == pytest.approx(5.0)


def test_custom_workflow_sequence(monkeypatch: pytest.MonkeyPatch, simple_config: dict) -> None:
    config = copy.deepcopy(simple_config)
    config["scenario"] = {
        "workflow": ["PF", "RH"],
        "fix_design": False,
        "rolling_horizon": {"HEAT_HORIZON_HOURS": 4.0, "STEP_HOURS": 2.0},
    }

    table = _make_table(4)
    monkeypatch.setattr(rh, "load_input_excel", lambda *args, **kwargs: table)

    pf_summary = OrderedDict(
        {
            "objective": OrderedDict(),
            "heat_pump_HP1": OrderedDict(
                [
                    ("Thermal_capacity_MW", 5.0),
                    ("Build_binary", 1.0),
                ]
            ),
        }
    )

    call_state = {"idx": 0}

    def fake_solve(table_arg, cfg, dt_h, solver_name):
        idx = call_state["idx"]
        call_state["idx"] += 1
        if idx == 0:
            series = OrderedDict({"TES_SOC_MWh": [0.0] * len(table_arg)})
            return rh.ScenarioResult(table_arg, series, pf_summary, {}, {})
        hp_cfg = cfg["system"]["heat_pumps"][0]
        # Design fixation is disabled
        assert hp_cfg.get("investment", {}).get("enabled", True) is True
        series = OrderedDict({"TES_SOC_MWh": [0.0] * len(table_arg)})
        return rh.ScenarioResult(table_arg, series, {}, {}, {})

    monkeypatch.setattr(rh, "_solve_scenario", fake_solve)

    result = rh.run_workflow([], overrides=config)

    assert result.pf_result is not None
    assert result.rh_result is not None
    assert result.plan.steps == ["PF", "RH"]


def test_workflow_accepts_string(monkeypatch: pytest.MonkeyPatch, simple_config: dict) -> None:
    config = copy.deepcopy(simple_config)
    config["scenario"] = {"workflow": "PF"}

    table = _make_table(2)
    monkeypatch.setattr(rh, "load_input_excel", lambda *args, **kwargs: table)

    def fake_solve(table_arg, cfg, dt_h, solver_name):
        series = OrderedDict({"TES_SOC_MWh": [0.0] * len(table_arg)})
        summary = OrderedDict({"objective": OrderedDict()})
        return rh.ScenarioResult(table_arg, series, summary, {}, {})

    monkeypatch.setattr(rh, "_solve_scenario", fake_solve)

    result = rh.run_workflow([], overrides=config)
    assert result.pf_result is not None
    assert result.plan.steps == ["PF"]


def test_unknown_workflow_step(monkeypatch: pytest.MonkeyPatch, simple_config: dict) -> None:
    config = copy.deepcopy(simple_config)
    config["scenario"] = {"workflow": ["PF", "UNKNOWN"]}

    table = _make_table(2)
    monkeypatch.setattr(rh, "load_input_excel", lambda *args, **kwargs: table)

    def fake_solve(table_arg, cfg, dt_h, solver_name):
        series = OrderedDict({"TES_SOC_MWh": [0.0] * len(table_arg)})
        summary = OrderedDict({"objective": OrderedDict()})
        return rh.ScenarioResult(table_arg, series, summary, {}, {})

    monkeypatch.setattr(rh, "_solve_scenario", fake_solve)

    with pytest.raises(ValueError):
        rh.run_workflow([], overrides=config)


def test_custom_workflow_registration(monkeypatch: pytest.MonkeyPatch, simple_config: dict) -> None:
    config = copy.deepcopy(simple_config)
    config["scenario"] = {"workflow": ["CUSTOM"]}

    table = _make_table(2)
    monkeypatch.setattr(rh, "load_input_excel", lambda *args, **kwargs: table)

    def fake_merge(paths):
        return copy.deepcopy(config)

    monkeypatch.setattr(rh, "load_and_merge", fake_merge)

    executed: List[str] = []

    def custom_step(context: rh.WorkflowContext) -> None:
        executed.append("X")
        context.pf_result = rh.ScenarioResult(context.table, OrderedDict(), {}, {}, {})

    rh.register_workflow_step("CUSTOM", custom_step)
    try:
        result = rh.run_workflow(["dummy.yaml"])
    finally:
        rh.unregister_workflow_step("CUSTOM")

    assert executed == ["X"]
    assert result.plan.steps == ["CUSTOM"]
    assert result.pf_result is not None


def test_cli_entrypoint(monkeypatch: pytest.MonkeyPatch, simple_config: dict, capsys: pytest.CaptureFixture[str]) -> None:
    config = copy.deepcopy(simple_config)
    config["scenario"] = {"workflow": ["PF"]}

    table = _make_table(2)

    def fake_merge(paths):
        return copy.deepcopy(config)

    def fake_solve(table_arg, cfg, dt_h, solver_name):
        series = OrderedDict({"TES_SOC_MWh": [0.0] * len(table_arg)})
        summary = OrderedDict({"objective": OrderedDict()})
        return rh.ScenarioResult(table_arg, series, summary, {"objective.OBJ_value_EUR": 0.0}, {})

    monkeypatch.setattr(rh, "load_and_merge", fake_merge)
    monkeypatch.setattr(rh, "load_input_excel", lambda *args, **kwargs: table)
    monkeypatch.setattr(rh, "_solve_scenario", fake_solve)

    exit_code = rh.main(["configs.yaml", "--print-design"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "[workflow] Executed steps" in captured.out

