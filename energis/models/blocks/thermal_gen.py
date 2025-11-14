from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import pyomo.environ as pyo
except Exception:  # pragma: no cover
    pyo = None

class ThermalGeneratorBlock:
    def __init__(self, name: str, th_eff: float, el_eff: float | None, cap_th_mw: float):
        self.name = name
        self.th_eff = float(th_eff)
        self.el_eff = None if el_eff is None else float(el_eff)
        self.cap_th = float(cap_th_mw)

    def attach(self, m, Tset, cfg, buses):
        if pyo is None:
            raise RuntimeError("Pyomo is required to attach blocks")
        comp = self.name
        setattr(m, f"{comp}_Qth", pyo.Var(Tset, domain=pyo.NonNegativeReals))
        setattr(m, f"{comp}_fuel", pyo.Var(Tset, domain=pyo.NonNegativeReals))
        Qth = getattr(m, f"{comp}_Qth")
        fuel= getattr(m, f"{comp}_fuel")

        setattr(m, f"{comp}_Qmax", pyo.Param(initialize=self.cap_th))
        setattr(m, f"{comp}_th_eff", pyo.Param(initialize=self.th_eff))

        def cap_rule(mm, t): return Qth[t] <= mm.__getattribute__(f"{comp}_Qmax")
        def th_link(mm, t):  return Qth[t] == mm.__getattribute__(f"{comp}_th_eff") * fuel[t]
        setattr(m, f"{comp}_cap", pyo.Constraint(Tset, rule=cap_rule))
        setattr(m, f"{comp}_thlink", pyo.Constraint(Tset, rule=th_link))

        Pel = None
        if self.el_eff is not None:
            setattr(m, f"{comp}_el_eff", pyo.Param(initialize=self.el_eff))
            setattr(m, f"{comp}_Pel", pyo.Var(Tset, domain=pyo.NonNegativeReals))
            Pelv = getattr(m, f"{comp}_Pel")
            def el_link(mm, t): return Pelv[t] == mm.__getattribute__(f"{comp}_el_eff") * fuel[t]
            setattr(m, f"{comp}_ellink", pyo.Constraint(Tset, rule=el_link))
            Pel = Pelv

        return {"Q_th_out": Qth, "fuel_in": fuel, "P_el_out": Pel}

