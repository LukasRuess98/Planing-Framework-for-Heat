import pytest
from energis.config.merge import load_and_merge
from energis.io.loader import load_input_excel
import pandas as pd

def test_merge_and_loader_smoke(tmp_path):
    # minimal merge smoke test
    cfg = load_and_merge([
        "configs/base.yaml",
        "configs/tech_catalog.yaml",
        "configs/sites/default.site.yaml",
        "configs/systems/baseline.system.yaml",
        "configs/scenarios/pf_then_rh.scenario.yaml",
    ])
    assert "site" in cfg and "system" in cfg

    # synthesize tiny excel to check DST pipeline (no pyomo needed)
    xls = tmp_path/"Import_Data.xlsx"
    idx = pd.date_range("2023-03-26 00:00", periods=6, freq="H")  # EU DST change happens around this date
    df = pd.DataFrame({
        "Datum": idx,
        "Day_Ahead_Price €/MWh": [50,52,48,47,49,55],
        "Wärmebedarf MW": [10,10,10,10,10,10],
        "CO2_consumption_based kgCO2/MWh": [350,360,340,330,335,345],
        "WRG1Q MW":[2,2,2,2,2,2],
        "WRG1_T °C":[50,50,50,50,50,50],
        "WRG2Q MW":[1,1,1,1,1,1],
        "WRG2_T °C":[45,45,45,45,45,45],
        "WRG3Q MW":[0,0,0,0,0,0],
        "WRG3_T °C":[40,40,40,40,40,40],
        "WRG4Q MW":[0,0,0,0,0,0],
        "WRG4_T °C":[35,35,35,35,35,35],
    })
    with pd.ExcelWriter(xls) as xw:
        df.to_excel(xw, index=False)

    cfg["site"]["input_xlsx"] = str(xls)
    data = load_input_excel(cfg["site"]["input_xlsx"], cfg["site"], dt_hours=1.0)
    assert len(data) == 6
    assert {"strompreis_EUR_MWh","waermebedarf_MWth","grid_co2_kg_MWh"}.issubset(set(data.columns))
