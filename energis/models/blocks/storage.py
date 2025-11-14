from __future__ import annotations
import pyomo.environ as pyo

class StorageBlock:
    def __init__(self, name: str, e_min, e_max, p_max, eff_c, eff_d, hourly_loss, dt_h, soc0):
        self.name = name
        self.e_min = float(e_min); self.e_max = float(e_max)
        self.p_max = float(p_max)
        self.eff_c = float(eff_c); self.eff_d = float(eff_d)
        self.hourly_loss = float(hourly_loss)
        self.dt_h = float(dt_h)
        self.soc0 = float(soc0)

    def attach(self, m, Tset, cfg, buses):
        comp = self.name
        setattr(m, f"{comp}_E", pyo.Var(Tset, domain=pyo.NonNegativeReals))
        setattr(m, f"{comp}_Qc", pyo.Var(Tset, domain=pyo.NonNegativeReals))  # charge heat in
        setattr(m, f"{comp}_Qd", pyo.Var(Tset, domain=pyo.NonNegativeReals))  # discharge heat out

        E  = getattr(m, f"{comp}_E")
        Qc = getattr(m, f"{comp}_Qc")
        Qd = getattr(m, f"{comp}_Qd")

        setattr(m, f"{comp}_Emin", pyo.Param(initialize=self.e_min))
        setattr(m, f"{comp}_Emax", pyo.Param(initialize=self.e_max))
        setattr(m, f"{comp}_Pmax", pyo.Param(initialize=self.p_max))
        setattr(m, f"{comp}_effc", pyo.Param(initialize=self.eff_c))
        setattr(m, f"{comp}_effd", pyo.Param(initialize=self.eff_d))
        setattr(m, f"{comp}_loss", pyo.Param(initialize=self.hourly_loss ** self.dt_h))

        def ecap_hi(mm, t): return E[t] <= mm.__getattribute__(f"{comp}_Emax")
        def ecap_lo(mm, t): return E[t] >= mm.__getattribute__(f"{comp}_Emin")
        def pcap_c(mm, t):  return Qc[t] <= mm.__getattribute__(f"{comp}_Pmax")
        def pcap_d(mm, t):  return Qd[t] <= mm.__getattribute__(f"{comp}_Pmax")

        setattr(m, f"{comp}_ecap_hi", pyo.Constraint(Tset, rule=ecap_hi))
        setattr(m, f"{comp}_ecap_lo", pyo.Constraint(Tset, rule=ecap_lo))
        setattr(m, f"{comp}_pcap_c",  pyo.Constraint(Tset, rule=pcap_c))
        setattr(m, f"{comp}_pcap_d",  pyo.Constraint(Tset, rule=pcap_d))

        def soc_dyn(mm, t):
            prev = E[t-1] * mm.__getattribute__(f"{comp}_loss") if t > Tset.first() else self.soc0
            return E[t] == prev + mm.__getattribute__(f"{comp}_effc") * Qc[t] * self.dt_h - (1.0/mm.__getattribute__(f"{comp}_effd")) * Qd[t] * self.dt_h
        setattr(m, f"{comp}_soc", pyo.Constraint(Tset, rule=soc_dyn))

        return {"Q_th_out": Qd, "Q_th_in": Qc, "SOC": E}
