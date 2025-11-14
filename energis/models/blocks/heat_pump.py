from __future__ import annotations

from typing import Dict, Any, List, Optional

try:  # pragma: no cover - optional dependency
    import pyomo.environ as pyo
except Exception:  # pragma: no cover
    pyo = None

class HeatPumpBlock:
    def __init__(
        self,
        name: str,
        min_load: float,
        cop_series: List[float],
        *,
        capacity_min_mw: float,
        capacity_max_mw: float,
        capacity_init_mw: float,
        investable: bool,
        wrg_cap_series: Optional[Dict[int, float]] = None,
    ):
        self.name = name
        self.min_load = float(min_load)
        self.COP_series = [float(c) for c in cop_series]
        self.capacity_min_mw = float(capacity_min_mw)
        self.capacity_max_mw = float(capacity_max_mw)
        self.capacity_init_mw = float(capacity_init_mw)
        self.investable = bool(investable)
        self.wrg_cap_series = wrg_cap_series or {}

    def attach(self, m, Tset, cfg, buses):
        if pyo is None:
            raise RuntimeError("Pyomo is required to attach blocks")
        # Create variables
        comp = self.name
        setattr(m, f"{comp}_Q", pyo.Var(Tset, domain=pyo.NonNegativeReals))  # heat out
        setattr(m, f"{comp}_Pel", pyo.Var(Tset, domain=pyo.NonNegativeReals)) # electricity in
        setattr(m, f"{comp}_on", pyo.Var(Tset, domain=pyo.Binary))
        setattr(m, f"{comp}_build", pyo.Var(domain=pyo.Binary))
        setattr(
            m,
            f"{comp}_cap_mw",
            pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, self.capacity_max_mw)),
        )

        Q = getattr(m, f"{comp}_Q")
        Pel = getattr(m, f"{comp}_Pel")
        onv = getattr(m, f"{comp}_on")
        build = getattr(m, f"{comp}_build")
        cap = getattr(m, f"{comp}_cap_mw")

        # Params
        setattr(m, f"{comp}_minload", pyo.Param(initialize=self.min_load))
        COPp = pyo.Param(Tset, initialize={t: self.COP_series[t-1] for t in Tset}, mutable=False)
        setattr(m, f"{comp}_COP", COPp)

        if not self.investable:
            build.fix(1 if self.capacity_init_mw > 0 else 0)
            cap.fix(self.capacity_init_mw)
        else:
            cap.set_value(self.capacity_init_mw)

        setattr(m, f"{comp}_cap_min", pyo.Param(initialize=self.capacity_min_mw))
        setattr(m, f"{comp}_cap_max", pyo.Param(initialize=self.capacity_max_mw))

        # Constraints
        def cap_rule(mm, t):
            return Q[t] <= cap * onv[t]
        setattr(m, f"{comp}_cap", pyo.Constraint(Tset, rule=cap_rule))

        def min_rule(mm, t):
            return Q[t] >= mm.__getattribute__(f"{comp}_minload") * cap * onv[t]
        setattr(m, f"{comp}_min", pyo.Constraint(Tset, rule=min_rule))

        def pel_link(mm, t):
            return Pel[t] == Q[t] / mm.__getattribute__(f"{comp}_COP")[t]
        setattr(m, f"{comp}_el_link", pyo.Constraint(Tset, rule=pel_link))

        def cap_hi(mm):
            return cap <= mm.__getattribute__(f"{comp}_cap_max") * build
        setattr(m, f"{comp}_cap_hi", pyo.Constraint(rule=cap_hi))

        def cap_lo(mm):
            return cap >= mm.__getattribute__(f"{comp}_cap_min") * build
        setattr(m, f"{comp}_cap_lo", pyo.Constraint(rule=cap_lo))

        def mode_link(mm, t):
            return onv[t] <= build
        setattr(m, f"{comp}_mode_link", pyo.Constraint(Tset, rule=mode_link))

        if self.wrg_cap_series:
            wrg_param = pyo.Param(Tset, initialize=self.wrg_cap_series, mutable=False)
            setattr(m, f"{comp}_WRG_cap", wrg_param)

            def wrg_rule(mm, t):
                return Q[t] <= wrg_param[t]

            setattr(m, f"{comp}_wrg_limit", pyo.Constraint(Tset, rule=wrg_rule))

        # Flows to buses
        return {
            "Q_th_out": Q,
            "P_el_in": Pel,
            "build": build,
            "capacity": cap,
        }

