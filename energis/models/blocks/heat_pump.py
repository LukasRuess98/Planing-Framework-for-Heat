from __future__ import annotations
from typing import Dict, Any, List
import pyomo.environ as pyo

class HeatPumpBlock:
    def __init__(self, name: str, max_th_mw: float, min_load: float, COP_series: List[float]):
        self.name = name
        self.max_th_mw = float(max_th_mw)
        self.min_load = float(min_load)
        self.COP_series = [float(c) for c in COP_series]

    def attach(self, m, Tset, cfg, buses):
        # Create variables
        comp = self.name
        setattr(m, f"{comp}_Q", pyo.Var(Tset, domain=pyo.NonNegativeReals))  # heat out
        setattr(m, f"{comp}_Pel", pyo.Var(Tset, domain=pyo.NonNegativeReals)) # electricity in
        setattr(m, f"{comp}_on", pyo.Var(Tset, domain=pyo.Binary))

        Q = getattr(m, f"{comp}_Q")
        Pel = getattr(m, f"{comp}_Pel")
        onv = getattr(m, f"{comp}_on")

        # Params
        setattr(m, f"{comp}_Qmax", pyo.Param(initialize=self.max_th_mw))
        setattr(m, f"{comp}_minload", pyo.Param(initialize=self.min_load))
        COPp = pyo.Param(Tset, initialize={t: self.COP_series[t-1] for t in Tset}, mutable=False)
        setattr(m, f"{comp}_COP", COPp)

        # Constraints
        def cap_rule(mm, t):
            return Q[t] <= mm.__getattribute__(f"{comp}_Qmax") * onv[t]
        setattr(m, f"{comp}_cap", pyo.Constraint(Tset, rule=cap_rule))

        def min_rule(mm, t):
            return Q[t] >= mm.__getattribute__(f"{comp}_minload") * mm.__getattribute__(f"{comp}_Qmax") * onv[t]
        setattr(m, f"{comp}_min", pyo.Constraint(Tset, rule=min_rule))

        def pel_link(mm, t):
            return Pel[t] == Q[t] / mm.__getattribute__(f"{comp}_COP")[t]
        setattr(m, f"{comp}_el_link", pyo.Constraint(Tset, rule=pel_link))

        # Flows to buses
        return {"Q_th_out": Q, "P_el_in": Pel}