from __future__ import annotations
from typing import Dict, Any, List
import pyomo.environ as pyo
import numpy as np

def _lmtd(Th, Tc):
    Th = float(Th); Tc = float(Tc)
    # Avoid zero division or negative inside log
    d1 = max(Th - Tc, 1e-6)
    return d1 / max(abs(np.log(max(Th-1e-9, Th)/max(Tc+1e-9, Tc))), 1e-6)

def _cop_physics(Tsrc_in_K: float, Tsrc_out_K: float, sink_out_K: float,
                 sink_in_K: float, deltaTpp: float, eta: float, FQ: float) -> float:
    LMTD_sink = _lmtd(sink_out_K, sink_in_K)
    LMTD_source = _lmtd(Tsrc_in_K, Tsrc_out_K)
    mdts = 0.2*(sink_out_K - Tsrc_out_K + 2*deltaTpp) + 0.2*(sink_out_K - sink_in_K) + 0.016
    qww  = 0.0014*(sink_out_K - Tsrc_out_K + 2*deltaTpp) - 0.0015*(sink_out_K - sink_in_K) + 0.039
    A = LMTD_sink / max(1e-9, (LMTD_sink - LMTD_source))
    B = (1 + (mdts + deltaTpp)/max(1e-9,LMTD_sink)) / (1 + (mdts + 0.5*(Tsrc_in_K - Tsrc_out_K) + 2*deltaTpp)/max(1e-9,(LMTD_sink - LMTD_source)))
    COP = A * B * eta * (1 - qww) + 1 - eta - FQ
    return float(np.clip(COP if np.isfinite(COP) else 3.0, 1.01, 12.0))

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
