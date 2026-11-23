from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from .data import NetworkInstance
from .model import build_lp


@dataclass
class MilpOutput:
    result: object
    solution: pd.DataFrame
    elapsed_s: float


def solve_integer_instance(instance: NetworkInstance, presolve: bool = True) -> MilpOutput:
    c, A_ub, b_ub, A_eq, b_eq, allowed_routes = build_lp(instance)
    n = len(c)
    constraints = [
        LinearConstraint(A_ub, -np.inf, b_ub),
        LinearConstraint(A_eq, b_eq, b_eq),
    ]
    bounds = Bounds(np.zeros(n, dtype=float), np.full(n, np.inf, dtype=float))
    integrality = np.ones(n, dtype=int)

    t0 = time.perf_counter()
    res = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={"presolve": presolve},
    )
    elapsed = time.perf_counter() - t0

    if not res.success:
        return MilpOutput(res, pd.DataFrame(), elapsed)

    flows = allowed_routes.copy()
    flows["flow_units"] = np.asarray(res.x, dtype=float)
    flows["route_cost"] = flows["flow_units"] * flows["unit_cost"]

    all_routes = instance.routes.merge(
        flows[["warehouse_id", "store_id", "flow_units", "route_cost"]],
        on=["warehouse_id", "store_id"],
        how="left",
    )
    all_routes[["flow_units", "route_cost"]] = all_routes[["flow_units", "route_cost"]].fillna(0.0)
    return MilpOutput(res, all_routes, elapsed)


def save_milp_log(output: MilpOutput, path: Path, scipy_version: str, highs_version: str, presolve: bool = True) -> None:
    res = output.result
    lines = [
        "HiGHS MILP solver log summary",
        "solver=scipy.optimize.milp",
        f"scipy_version={scipy_version}",
        f"highs_version={highs_version}",
        f"presolve={presolve}",
        f"status_code={getattr(res, 'status', None)}",
        f"success={getattr(res, 'success', None)}",
        f"message={getattr(res, 'message', '')}",
        f"objective={getattr(res, 'fun', None)}",
        f"mip_node_count={getattr(res, 'mip_node_count', None)}",
        f"mip_gap={getattr(res, 'mip_gap', None)}",
        f"elapsed_s={output.elapsed_s:.9f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
