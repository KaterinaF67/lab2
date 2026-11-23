from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .data import NetworkInstance


@dataclass
class SolveOutput:
    result: object
    solution: pd.DataFrame
    warehouse_summary: pd.DataFrame
    store_summary: pd.DataFrame
    route_summary: pd.DataFrame
    bottlenecks: pd.DataFrame
    node_potentials: pd.DataFrame
    elapsed_s: float
    method: str


def _status_name(result) -> str:
    if getattr(result, "success", False):
        return "optimal"
    status = int(getattr(result, "status", -1))
    return {1: "iteration_limit", 2: "infeasible", 3: "unbounded", 4: "numerical_issue"}.get(status, "failed")


def build_lp(instance: NetworkInstance):
    allowed_routes = instance.routes[instance.routes["allowed"]].reset_index(drop=True)
    I = len(instance.warehouses)
    J = len(instance.stores)
    n = len(allowed_routes)

    wmap = {w: i for i, w in enumerate(instance.warehouses["warehouse_id"])}
    smap = {s: j for j, s in enumerate(instance.stores["store_id"])}

    c = allowed_routes["unit_cost"].to_numpy(dtype=float)
    A_eq = lil_matrix((I + J, n), dtype=float)
    b_eq = np.concatenate([
        instance.warehouses["supply_units"].to_numpy(dtype=float),
        instance.stores["demand_units"].to_numpy(dtype=float),
    ])

    for k, row in allowed_routes.iterrows():
        i = wmap[row["warehouse_id"]]
        j = smap[row["store_id"]]
        A_eq[i, k] = 1.0
        A_eq[I + j, k] = 1.0

    A_ub = lil_matrix((n, n), dtype=float)
    b_ub = allowed_routes["capacity_units"].to_numpy(dtype=float)
    for k in range(n):
        A_ub[k, k] = 1.0

    return c, A_ub.tocsr(), b_ub, A_eq.tocsr(), b_eq, allowed_routes


def solve_instance(instance: NetworkInstance, method: str = "highs-ds", presolve: bool = True, disp: bool = False) -> SolveOutput:
    c, A_ub, b_ub, A_eq, b_eq, allowed_routes = build_lp(instance)
    t0 = time.perf_counter()
    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=(0, None),
        method=method,
        options={"presolve": presolve, "disp": disp},
    )
    elapsed = time.perf_counter() - t0

    if not res.success:
        return SolveOutput(
            res,
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            elapsed,
            method,
        )

    flows = allowed_routes.copy()
    flows["flow_units"] = np.asarray(res.x, dtype=float)
    flows["route_cost"] = flows["flow_units"] * flows["unit_cost"]
    cap_residual = np.asarray(res.ineqlin.residual, dtype=float)
    cap_marginal = np.asarray(res.ineqlin.marginals, dtype=float)
    flows["capacity_slack"] = cap_residual
    flows["capacity_utilization"] = np.where(
        flows["capacity_units"] > 0,
        flows["flow_units"] / flows["capacity_units"],
        0.0,
    )
    flows["capacity_shadow_benefit"] = -cap_marginal

    all_routes = instance.routes.merge(
        flows[[
            "warehouse_id", "store_id", "flow_units", "route_cost", "capacity_slack",
            "capacity_utilization", "capacity_shadow_benefit"
        ]],
        on=["warehouse_id", "store_id"],
        how="left",
    )
    fill_cols = ["flow_units", "route_cost", "capacity_slack", "capacity_utilization", "capacity_shadow_benefit"]
    all_routes[fill_cols] = all_routes[fill_cols].fillna(0.0)

    wh_rows = []
    for row in instance.warehouses.itertuples(index=False):
        outflow = float(all_routes.loc[all_routes["warehouse_id"] == row.warehouse_id, "flow_units"].sum())
        wh_rows.append({
            "warehouse_id": row.warehouse_id,
            "supply_units": float(row.supply_units),
            "outflow_units": outflow,
            "balance_error": outflow - float(row.supply_units),
        })
    warehouse_summary = pd.DataFrame(wh_rows)

    st_rows = []
    for row in instance.stores.itertuples(index=False):
        inflow = float(all_routes.loc[all_routes["store_id"] == row.store_id, "flow_units"].sum())
        st_rows.append({
            "store_id": row.store_id,
            "demand_units": float(row.demand_units),
            "inflow_units": inflow,
            "balance_error": inflow - float(row.demand_units),
        })
    store_summary = pd.DataFrame(st_rows)

    bottlenecks = flows[(flows["capacity_slack"] <= 1e-7) & (flows["flow_units"] > 1e-7)].copy()
    bottlenecks = bottlenecks.sort_values("capacity_shadow_benefit", ascending=False).reset_index(drop=True)

    eq_marg = np.asarray(res.eqlin.marginals, dtype=float)
    I = len(instance.warehouses)
    potentials = []
    for i, wid in enumerate(instance.warehouses["warehouse_id"]):
        potentials.append({"node_id": wid, "node_type": "warehouse", "dual_potential": float(eq_marg[i])})
    for j, sid in enumerate(instance.stores["store_id"]):
        potentials.append({"node_id": sid, "node_type": "store", "dual_potential": float(eq_marg[I + j])})
    node_potentials = pd.DataFrame(potentials)

    return SolveOutput(
        res,
        all_routes,
        warehouse_summary,
        store_summary,
        all_routes.copy(),
        bottlenecks,
        node_potentials,
        elapsed,
        method,
    )


def save_solver_log(output: SolveOutput, path: Path, scipy_version: str, highs_version: str, presolve: bool = True) -> None:
    res = output.result
    lines = [
        "HiGHS solver log summary",
        f"method={output.method}",
        f"scipy_version={scipy_version}",
        f"highs_version={highs_version}",
        f"presolve={presolve}",
        f"status={_status_name(res)}",
        f"status_code={getattr(res, 'status', None)}",
        f"success={getattr(res, 'success', None)}",
        f"message={getattr(res, 'message', '')}",
        f"objective={getattr(res, 'fun', None)}",
        f"iterations={getattr(res, 'nit', None)}",
        f"crossover_iterations={getattr(res, 'crossover_nit', None)}",
        f"elapsed_s={output.elapsed_s:.9f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def status_name(result) -> str:
    return _status_name(result)
