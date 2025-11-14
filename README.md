# EnerGIS – Modular MILP Framework (Fuel Buses enabled)

This package contains a minimal, runnable skeleton to model industrial energy systems with **explicit fuel buses**.
It keeps the structure close to oemof/pypsa (configs, components, buses, orchestrator) and your previous monolithic code.

## Highlights
- YAML config-layer with merge order (base → tech_catalog → site → system → scenario → overrides.local)
- Explicit **buses**: `electricity`, `heat`, `gas`, `biomass`, `waste`
- Modular components (blocks): HeatPump, Storage, ThermalGenerator, P2H
- PF + RH orchestration and single Excel export
- Tests that build a tiny model and check bus balances (skipped if Pyomo missing)

## Quickstart
1. Put your `Import_Data.xlsx` in the repo root or change the path in `configs/sites/default.site.yaml`.
2. Install deps:
   ```bash
   pip install pyomo openpyxl pandas numpy pyyaml pytest
   # Optional solvers: gurobi / highs / glpk (fallback to glpk if available)
   ```
3. Run quick test:
   ```bash
   python quickstart_test.py
   ```
4. Or open `notebooks/01_scenario_studio.ipynb` and run.

Exports go to `exports/<timestamp>_<tag>/scenario.xlsx`.
