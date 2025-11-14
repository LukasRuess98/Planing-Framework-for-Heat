from __future__ import annotations

try:  # pragma: no cover - optional dependency
    import pyomo.environ as pyo
except Exception:  # pragma: no cover
    pyo = None

class StorageBlock:
    def __init__(
        self,
        name: str,
        e_min,
        e_max,
        p_max,
        eff_c,
        eff_d,
        hourly_loss,
        dt_h,
        soc0,
        *,
        investable: bool,
        e_cap_min: float,
        e_cap_max: float,
        p_cap_min: float,
        p_cap_max: float,
        e_cap_init: float,
        p_cap_init: float,
        terminal_target: float | None,
    ):
        self.name = name
        self.e_min = float(e_min); self.e_max = float(e_max)
        self.p_max = float(p_max)
        self.eff_c = float(eff_c); self.eff_d = float(eff_d)
        self.hourly_loss = float(hourly_loss)
        self.dt_h = float(dt_h)
        self.soc0 = float(soc0)
        self.investable = bool(investable)
        self.e_cap_min = float(e_cap_min)
        self.e_cap_max = float(e_cap_max)
        self.p_cap_min = float(p_cap_min)
        self.p_cap_max = float(p_cap_max)
        self.e_cap_init = float(e_cap_init)
        self.p_cap_init = float(p_cap_init)
        self.terminal_target = terminal_target

    def attach(self, m, Tset, cfg, buses):
        if pyo is None:
            raise RuntimeError("Pyomo is required to attach blocks")
        comp = self.name
        setattr(m, f"{comp}_E", pyo.Var(Tset, domain=pyo.NonNegativeReals))
        setattr(m, f"{comp}_Qc", pyo.Var(Tset, domain=pyo.NonNegativeReals))  # charge heat in
        setattr(m, f"{comp}_Qd", pyo.Var(Tset, domain=pyo.NonNegativeReals))  # discharge heat out
        setattr(m, f"{comp}_build", pyo.Var(domain=pyo.Binary))
        setattr(
            m,
            f"{comp}_cap_energy",
            pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, self.e_cap_max)),
        )
        setattr(
            m,
            f"{comp}_cap_power",
            pyo.Var(domain=pyo.NonNegativeReals, bounds=(0.0, self.p_cap_max)),
        )

        E  = getattr(m, f"{comp}_E")
        Qc = getattr(m, f"{comp}_Qc")
        Qd = getattr(m, f"{comp}_Qd")
        build = getattr(m, f"{comp}_build")
        cap_e = getattr(m, f"{comp}_cap_energy")
        cap_p = getattr(m, f"{comp}_cap_power")

        setattr(m, f"{comp}_Emin", pyo.Param(initialize=self.e_min))
        setattr(m, f"{comp}_Emax", pyo.Param(initialize=self.e_max))
        setattr(m, f"{comp}_Pmax", pyo.Param(initialize=self.p_max))
        setattr(m, f"{comp}_effc", pyo.Param(initialize=self.eff_c))
        setattr(m, f"{comp}_effd", pyo.Param(initialize=self.eff_d))
        setattr(m, f"{comp}_loss", pyo.Param(initialize=self.hourly_loss ** self.dt_h))
        setattr(m, f"{comp}_capE_min", pyo.Param(initialize=self.e_cap_min))
        setattr(m, f"{comp}_capE_max", pyo.Param(initialize=self.e_cap_max))
        setattr(m, f"{comp}_capP_min", pyo.Param(initialize=self.p_cap_min))
        setattr(m, f"{comp}_capP_max", pyo.Param(initialize=self.p_cap_max))

        if not self.investable:
            build.fix(1 if self.e_cap_max > 0 else 0)
            cap_e.fix(self.e_cap_init if self.e_cap_init > 0 else self.e_cap_max)
            cap_p.fix(self.p_cap_init if self.p_cap_init > 0 else self.p_cap_max)
        else:
            cap_e.set_value(self.e_cap_init if self.e_cap_init > 0 else self.e_cap_min)
            cap_p.set_value(self.p_cap_init if self.p_cap_init > 0 else self.p_cap_min)

        def ecap_hi(mm, t): return E[t] <= cap_e
        def ecap_lo(mm, t): return E[t] >= mm.__getattribute__(f"{comp}_Emin") * build
        def pcap_c(mm, t):  return Qc[t] <= cap_p
        def pcap_d(mm, t):  return Qd[t] <= cap_p

        setattr(m, f"{comp}_ecap_hi", pyo.Constraint(Tset, rule=ecap_hi))
        setattr(m, f"{comp}_ecap_lo", pyo.Constraint(Tset, rule=ecap_lo))
        setattr(m, f"{comp}_pcap_c",  pyo.Constraint(Tset, rule=pcap_c))
        setattr(m, f"{comp}_pcap_d",  pyo.Constraint(Tset, rule=pcap_d))

        def cap_e_hi(mm):
            return cap_e <= mm.__getattribute__(f"{comp}_capE_max") * build
        setattr(m, f"{comp}_capE_hi", pyo.Constraint(rule=cap_e_hi))

        def cap_e_lo(mm):
            return cap_e >= mm.__getattribute__(f"{comp}_capE_min") * build
        setattr(m, f"{comp}_capE_lo", pyo.Constraint(rule=cap_e_lo))

        def cap_p_hi(mm):
            return cap_p <= mm.__getattribute__(f"{comp}_capP_max") * build
        setattr(m, f"{comp}_capP_hi", pyo.Constraint(rule=cap_p_hi))

        def cap_p_lo(mm):
            return cap_p >= mm.__getattribute__(f"{comp}_capP_min") * build
        setattr(m, f"{comp}_capP_lo", pyo.Constraint(rule=cap_p_lo))

        def soc_dyn(mm, t):
            prev = E[t-1] * mm.__getattribute__(f"{comp}_loss") if t > Tset.first() else self.soc0
            return E[t] == prev + mm.__getattribute__(f"{comp}_effc") * Qc[t] * self.dt_h - (1.0/mm.__getattribute__(f"{comp}_effd")) * Qd[t] * self.dt_h
        setattr(m, f"{comp}_soc", pyo.Constraint(Tset, rule=soc_dyn))

        if self.soc0 > 0:
            setattr(m, f"{comp}_soc0_cap", pyo.Constraint(expr=self.soc0 <= cap_e))

        if self.terminal_target is not None:
            target = float(self.terminal_target)
            setattr(m, f"{comp}_terminal", pyo.Constraint(expr=E[Tset.last()] == target))

        return {
            "Q_th_out": Qd,
            "Q_th_in": Qc,
            "SOC": E,
            "build": build,
            "cap_energy": cap_e,
            "cap_power": cap_p,
        }

