from __future__ import annotations
import pyomo.environ as pyo

class Bus:
    """Simple bus collector: each bus has a scalar balance equation per time step."""
    def __init__(self, name: str):
        self.name = name
        self.sources = []   # list of Var indexed by t (positive supply into bus)
        self.sinks = []     # list of Var indexed by t (demand out of bus)

    def add_source(self, var):  # var[t]
        self.sources.append(var)

    def add_sink(self, var):    # var[t]
        self.sinks.append(var)

    def attach_balance(self, m, tset):
        cname = f"bal_{self.name}"
        def _rule(mm, t):
            lhs = sum(v[t] for v in self.sources) if self.sources else 0
            rhs = sum(v[t] for v in self.sinks)   if self.sinks   else 0
            return lhs == rhs
        setattr(m, cname, pyo.Constraint(tset, rule=_rule))
        return getattr(m, cname)
