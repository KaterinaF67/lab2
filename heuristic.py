from __future__ import annotations

import numpy as np
import pandas as pd

from .data import NetworkInstance


def greedy_cheapest_routes(instance: NetworkInstance):
    supply = instance.warehouses.set_index("warehouse_id")["supply_units"].astype(float).to_dict()
    demand = instance.stores.set_index("store_id")["demand_units"].astype(float).to_dict()
    routes = instance.routes[instance.routes["allowed"]].sort_values(
        ["unit_cost", "warehouse_id", "store_id"]
    ).reset_index(drop=True)

    rows = []
    total_cost = 0.0
    for row in routes.itertuples(index=False):
        q = min(supply[row.warehouse_id], demand[row.store_id], float(row.capacity_units))
        q = max(0.0, float(q))
        if q > 0:
            supply[row.warehouse_id] -= q
            demand[row.store_id] -= q
            total_cost += q * float(row.unit_cost)
        rows.append({
            "warehouse_id": row.warehouse_id,
            "store_id": row.store_id,
            "unit_cost": float(row.unit_cost),
            "capacity_units": float(row.capacity_units),
            "flow_units": q,
            "route_cost": q * float(row.unit_cost),
        })

    feasible = max([abs(v) for v in supply.values()] + [abs(v) for v in demand.values()]) <= 1e-7
    solution = pd.DataFrame(rows)
    summary = {
        "feasible": bool(feasible),
        "objective_cost": float(total_cost) if feasible else None,
        "max_remaining_supply": float(max(abs(v) for v in supply.values())),
        "max_remaining_demand": float(max(abs(v) for v in demand.values())),
    }
    return solution, summary
