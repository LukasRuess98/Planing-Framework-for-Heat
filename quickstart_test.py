from energis.run.orchestrator import run_all

cfg_paths = [
    "configs/base.yaml",
    "configs/tech_catalog.yaml",
    "configs/sites/default.site.yaml",
    "configs/systems/baseline.system.yaml",
    "configs/scenarios/perfect_forecast_full_year.scenario.yaml",
    # "configs/overrides.local.yaml",  # optional
]

res = run_all(cfg_paths, overrides={"scenario": {"horizon": {"enforce": False}}})
print("Export:", res.get("scenario_xlsx"))
