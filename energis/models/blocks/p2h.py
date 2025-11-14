from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import pyomo.environ as pyo
except Exception:  # pragma: no cover
    pyo = None

class P2HBlock:
    def __init__(self, name: str, eff: float, cap_th_mw: float):
        self.name = name; self.eff = float(eff); self.cap = float(cap_th_mw)

    def attach(self, m, Tset, cfg, buses):
        if pyo is None:
            raise RuntimeError("Pyomo is required to attach blocks")
        comp = self.name
        setattr(m, f"{comp}_Qth", pyo.Var(Tset, domain=pyo.NonNegativeReals))
        setattr(m, f"{comp}_Pel", pyo.Var(Tset, domain=pyo.NonNegativeReals))
        Q = getattr(m, f"{comp}_Qth"); P = getattr(m, f"{comp}_Pel")
        setattr(m, f"{comp}_eff", pyo.Param(initialize=self.eff))
        setattr(m, f"{comp}_cap", pyo.Param(initialize=self.cap))

        def cap_rule(mm, t): return Q[t] <= mm.__getattribute__(f"{comp}_cap")
        def link(mm, t):     return Q[t] == mm.__getattribute__(f"{comp}_eff") * P[t]
        setattr(m, f"{comp}_capcons", pyo.Constraint(Tset, rule=cap_rule))
        setattr(m, f"{comp}_link",    pyo.Constraint(Tset, rule=link))
        return {"Q_th_out": Q, "P_el_in": P}

