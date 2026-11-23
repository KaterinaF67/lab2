from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from .data import NetworkInstance


def verify_from_csv(instance: NetworkInstance, solution_csv: Path, expected_objective: float | None = None, tol: float = 1e-7):
    sol = pd.read_csv(solution_csv)
    flows = sol[["warehouse_id", "store_id", "flow_units"]].copy()
    route_info = instance.routes[["warehouse_id", "store_id", "allowed", "capacity_units", "unit_cost"]]
    merged = route_info.merge(flows, on=["warehouse_id", "store_id"], how="left")
    merged["flow_units"] = merged["flow_units"].fillna(0.0)

    nonnegative_violation = max(0.0, -float(merged["flow_units"].min()))
    forbidden_violation = float(merged.loc[~merged["allowed"], "flow_units"].abs().max()) if (~merged["allowed"]).any() else 0.0
    capacity_violation = np.maximum(
        merged["flow_units"].to_numpy(dtype=float) - merged["capacity_units"].to_numpy(dtype=float),
        0.0,
    )

    wh = merged.groupby("warehouse_id", as_index=False)["flow_units"].sum()
    wh = instance.warehouses[["warehouse_id", "supply_units"]].merge(wh, on="warehouse_id", how="left").fillna(0.0)
    wh_error = wh["flow_units"].to_numpy(dtype=float) - wh["supply_units"].to_numpy(dtype=float)

    st = merged.groupby("store_id", as_index=False)["flow_units"].sum()
    st = instance.stores[["store_id", "demand_units"]].merge(st, on="store_id", how="left").fillna(0.0)
    st_error = st["flow_units"].to_numpy(dtype=float) - st["demand_units"].to_numpy(dtype=float)

    objective = float(np.sum(merged["flow_units"].to_numpy(dtype=float) * merged["unit_cost"].to_numpy(dtype=float)))
    objective_difference = None if expected_objective is None else abs(objective - expected_objective)

    report = {
        "tolerance": tol,
        "nonnegative_violation": nonnegative_violation,
        "forbidden_route_violation": forbidden_violation,
        "max_capacity_violation": float(capacity_violation.max(initial=0.0)),
        "max_warehouse_balance_error": float(np.max(np.abs(wh_error), initial=0.0)),
        "max_store_balance_error": float(np.max(np.abs(st_error), initial=0.0)),
        "objective_recomputed": objective,
        "objective_difference": objective_difference,
        "feasible": bool(
            nonnegative_violation <= tol
            and forbidden_violation <= tol
            and capacity_violation.max(initial=0.0) <= tol
            and np.max(np.abs(wh_error), initial=0.0) <= tol
            and np.max(np.abs(st_error), initial=0.0) <= tol
            and (objective_difference is None or objective_difference <= 1e-6)
        ),
    }
    return report


def save_verification(report: dict, path: Path) -> None:
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def verify_integer_from_csv(instance: NetworkInstance, solution_csv: Path, expected_objective: float | None = None, tol: float = 1e-7):
    report = verify_from_csv(instance, solution_csv, expected_objective=expected_objective, tol=tol)
    sol = pd.read_csv(solution_csv)
    values = sol["flow_units"].to_numpy(dtype=float)
    integrality_violation = float(np.max(np.abs(values - np.round(values)), initial=0.0))
    report["max_integrality_violation"] = integrality_violation
    report["feasible"] = bool(report["feasible"] and integrality_violation <= tol)
    return report
