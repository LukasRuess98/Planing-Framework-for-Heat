from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Dict, Iterable

try:  # pragma: no cover - optional dependency
    import pyomo.environ as pyo
except Exception:  # pragma: no cover
    pyo = None


def _prepare_series(
    indices: Iterable[int],
    series: Sequence[float] | Mapping[int, float] | None,
    default: float,
) -> Dict[int, float]:
    """Return mapping for Pyomo params while supporting broadcasting."""

    idx_list = list(indices)
    if series is None:
        return {i: float(default) for i in idx_list}
    if isinstance(series, Mapping):
        return {i: float(series.get(i, default)) for i in idx_list}
    values = [float(v) for v in series]
    if len(values) == 1:
        return {i: float(values[0]) for i in idx_list}
    if len(values) != len(idx_list):
        raise ValueError("Series length does not match time index length")
    return {i: float(values[pos]) for pos, i in enumerate(idx_list)}


def _clamp_positive(values: Dict[int, float], floor: float = 1e-6) -> Dict[int, float]:
    return {idx: (val if val > floor else floor) for idx, val in values.items()}


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
        loss_series: Sequence[float] | Mapping[int, float] | None = None,
        eff_charge_series: Sequence[float] | Mapping[int, float] | None = None,
        eff_discharge_series: Sequence[float] | Mapping[int, float] | None = None,
        capacity_active_series: Sequence[float] | Mapping[int, float] | None = None,
    ):
        self.name = name
        self.e_min = float(e_min)
        self.e_max = float(e_max)
        self.p_max = float(p_max)
        self.eff_c = float(eff_c)
        self.eff_d = float(eff_d)
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
        self.loss_series = loss_series
        self.eff_charge_series = eff_charge_series
        self.eff_discharge_series = eff_discharge_series
        self.capacity_active_series = capacity_active_series

    def attach(self, m, Tset, cfg, buses):
        if pyo is None:
            raise RuntimeError("Pyomo is required to attach blocks")
        comp = self.name
        setattr(m, f"{comp}_E", pyo.Var(Tset, domain=pyo.NonNegativeReals))
        setattr(m, f"{comp}_Qc", pyo.Var(Tset, domain=pyo.NonNegativeReals))  # charge heat in
        setattr(m, f"{comp}_Qd", pyo.Var(Tset, domain=pyo.NonNegativeReals))  # discharge heat out
        setattr(m, f"{comp}_build", pyo.Var(domain=pyo.Binary))
        setattr(m, f"{comp}_charge_mode", pyo.Var(Tset, domain=pyo.Binary))
        setattr(m, f"{comp}_discharge_mode", pyo.Var(Tset, domain=pyo.Binary))
        setattr(m, f"{comp}_active", pyo.Var(Tset, domain=pyo.Binary))
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

        E = getattr(m, f"{comp}_E")
        Qc = getattr(m, f"{comp}_Qc")
        Qd = getattr(m, f"{comp}_Qd")
        build = getattr(m, f"{comp}_build")
        charge_mode = getattr(m, f"{comp}_charge_mode")
        discharge_mode = getattr(m, f"{comp}_discharge_mode")
        active = getattr(m, f"{comp}_active")
        cap_e = getattr(m, f"{comp}_cap_energy")
        cap_p = getattr(m, f"{comp}_cap_power")

        times = list(Tset)
        loss_base = _prepare_series(times, self.loss_series, self.hourly_loss)
        loss_map = {idx: float(val) ** self.dt_h for idx, val in loss_base.items()}
        effc_map = _prepare_series(times, self.eff_charge_series, self.eff_c)
        effd_map = _clamp_positive(_prepare_series(times, self.eff_discharge_series, self.eff_d))
        active_limit_map = _prepare_series(times, self.capacity_active_series, 1.0)

        setattr(m, f"{comp}_Emin", pyo.Param(initialize=self.e_min))
        setattr(m, f"{comp}_Emax", pyo.Param(initialize=self.e_max))
        setattr(m, f"{comp}_Pmax", pyo.Param(initialize=self.p_max))
        setattr(m, f"{comp}_effc", pyo.Param(Tset, initialize=effc_map, mutable=True))
        setattr(m, f"{comp}_effd", pyo.Param(Tset, initialize=effd_map, mutable=True))
        setattr(m, f"{comp}_loss", pyo.Param(Tset, initialize=loss_map, mutable=True))
        setattr(
            m,
            f"{comp}_capacity_active_limit",
            pyo.Param(Tset, initialize=active_limit_map, mutable=True),
        )
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

        def ecap_hi(mm, t):
            return E[t] <= cap_e * active[t]

        def ecap_lo(mm, t):
            return E[t] >= mm.__getattribute__(f"{comp}_Emin") * active[t]

        def emax_hi(mm, t):
            return E[t] <= mm.__getattribute__(f"{comp}_Emax") * active[t]

        def pcap_c(mm, t):
            return Qc[t] <= cap_p * charge_mode[t]

        def pcap_d(mm, t):
            return Qd[t] <= cap_p * discharge_mode[t]

        setattr(m, f"{comp}_ecap_hi", pyo.Constraint(Tset, rule=ecap_hi))
        setattr(m, f"{comp}_ecap_lo", pyo.Constraint(Tset, rule=ecap_lo))
        setattr(m, f"{comp}_emax_hi", pyo.Constraint(Tset, rule=emax_hi))
        setattr(m, f"{comp}_pcap_c", pyo.Constraint(Tset, rule=pcap_c))
        setattr(m, f"{comp}_pcap_d", pyo.Constraint(Tset, rule=pcap_d))

        def active_build(mm, t):
            return active[t] <= build

        def active_limit(mm, t):
            return active[t] <= mm.__getattribute__(f"{comp}_capacity_active_limit")[t]

        def mode_cap(mm, t):
            return charge_mode[t] + discharge_mode[t] <= active[t]

        setattr(m, f"{comp}_active_build", pyo.Constraint(Tset, rule=active_build))
        setattr(m, f"{comp}_active_limit", pyo.Constraint(Tset, rule=active_limit))
        setattr(m, f"{comp}_mode_cap", pyo.Constraint(Tset, rule=mode_cap))

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
            loss = mm.__getattribute__(f"{comp}_loss")[t]
            effc = mm.__getattribute__(f"{comp}_effc")[t]
            effd = mm.__getattribute__(f"{comp}_effd")[t]
            prev_energy = E[t - 1] if t > Tset.first() else self.soc0
            return E[t] == prev_energy * loss + effc * Qc[t] * self.dt_h - (Qd[t] * self.dt_h) / effd

        setattr(m, f"{comp}_soc", pyo.Constraint(Tset, rule=soc_dyn))

        if self.soc0 > 0:
            setattr(m, f"{comp}_soc0_cap", pyo.Constraint(expr=self.soc0 <= cap_e))

        return {
            "Q_th_out": Qd,
            "Q_th_in": Qc,
            "SOC": E,
            "build": build,
            "cap_energy": cap_e,
            "cap_power": cap_p,
            "charge_mode": charge_mode,
            "discharge_mode": discharge_mode,
            "active": active,
        }
