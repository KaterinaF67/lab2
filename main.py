from __future__ import annotations

from pathlib import Path
import json
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy
import scipy.optimize._highspy._core as highs_core

from src.data import NetworkInstance, generate_instance, save_instance, load_instance, SEED, VARIANT
from src.model import solve_instance, save_solver_log, status_name
from src.verify import verify_from_csv, verify_integer_from_csv, save_verification
from src.heuristic import greedy_cheapest_routes
from src.plots import plot_network, plot_scenario_costs, plot_bottlenecks, plot_optimal_vs_greedy, plot_lp_vs_milp
from src.milp_extension import solve_integer_instance, save_milp_log

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
PLOTS = RESULTS / "plots"


def _copy_instance(instance: NetworkInstance) -> NetworkInstance:
    return NetworkInstance(instance.warehouses.copy(), instance.stores.copy(), instance.routes.copy())


def _save_solution(prefix: str, output) -> None:
    output.solution.to_csv(RESULTS / f"{prefix}_solution.csv", index=False, float_format="%.9f")
    output.warehouse_summary.to_csv(RESULTS / f"{prefix}_warehouse_check.csv", index=False, float_format="%.9f")
    output.store_summary.to_csv(RESULTS / f"{prefix}_store_check.csv", index=False, float_format="%.9f")
    output.route_summary.to_csv(RESULTS / f"{prefix}_route_check.csv", index=False, float_format="%.9f")


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    instance = generate_instance(seed=SEED)
    save_instance(instance, DATA)
    instance = load_instance(DATA)

    highs_version = f"{highs_core.HIGHS_VERSION_MAJOR}.{highs_core.HIGHS_VERSION_MINOR}.{highs_core.HIGHS_VERSION_PATCH}"

    base = solve_instance(instance, method="highs-ds", presolve=True)
    if not base.result.success:
        raise RuntimeError(f"Base solve failed: {base.result.message}")
    _save_solution("base", base)
    base.bottlenecks.to_csv(RESULTS / "bottlenecks.csv", index=False, float_format="%.9f")
    base.node_potentials.to_csv(RESULTS / "node_potentials.csv", index=False, float_format="%.9f")
    save_solver_log(base, RESULTS / "solver_base.log", scipy.__version__, highs_version, presolve=True)

    with (RESULTS / "solver_base_full.log").open("w", encoding="utf-8") as log_file:
        subprocess.run(
            [sys.executable, "-m", "src.highs_diagnostic", "--data", str(DATA), "--method", "highs-ds"],
            cwd=ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
        )

    verification_base = verify_from_csv(instance, RESULTS / "base_solution.csv", expected_objective=float(base.result.fun))
    save_verification(verification_base, RESULTS / "verification_base.json")
    if not verification_base["feasible"]:
        raise RuntimeError("Independent verification failed for base scenario")

    used = base.solution[base.solution["flow_units"] > 1e-7].copy()
    failed_row = used.sort_values(["flow_units", "warehouse_id", "store_id"], ascending=[False, True, True]).iloc[0]
    failed_route = (str(failed_row["warehouse_id"]), str(failed_row["store_id"]))

    failure_instance = _copy_instance(instance)
    mask = (
        (failure_instance.routes["warehouse_id"] == failed_route[0])
        & (failure_instance.routes["store_id"] == failed_route[1])
    )
    failure_instance.routes.loc[mask, "allowed"] = False
    failure_instance.routes.loc[mask, "capacity_units"] = 0.0
    failure = solve_instance(failure_instance, method="highs-ds", presolve=True)
    if not failure.result.success:
        raise RuntimeError(f"Arc-failure scenario failed unexpectedly: {failure.result.message}")
    _save_solution("failure", failure)
    save_solver_log(failure, RESULTS / "solver_failure.log", scipy.__version__, highs_version, presolve=True)
    verification_failure = verify_from_csv(
        failure_instance,
        RESULTS / "failure_solution.csv",
        expected_objective=float(failure.result.fun),
    )
    save_verification(verification_failure, RESULTS / "verification_failure.json")
    if not verification_failure["feasible"]:
        raise RuntimeError("Independent verification failed for arc-failure scenario")

    base_flow = base.solution[["warehouse_id", "store_id", "flow_units"]].rename(columns={"flow_units": "base_flow"})
    failure_flow = failure.solution[["warehouse_id", "store_id", "flow_units"]].rename(columns={"flow_units": "failure_flow"})
    changes = base_flow.merge(failure_flow, on=["warehouse_id", "store_id"], how="outer").fillna(0.0)
    changes["delta_flow"] = changes["failure_flow"] - changes["base_flow"]
    changes = changes[changes["delta_flow"].abs() > 1e-7].sort_values("delta_flow", key=np.abs, ascending=False)
    changes.to_csv(RESULTS / "failure_flow_changes.csv", index=False, float_format="%.9f")

    critical_bottleneck = base.bottlenecks.iloc[0]
    overload_route = (str(critical_bottleneck["warehouse_id"]), str(critical_bottleneck["store_id"]))
    overload_instance = _copy_instance(instance)
    overload_mask = (
        (overload_instance.routes["warehouse_id"] == overload_route[0])
        & (overload_instance.routes["store_id"] == overload_route[1])
    )
    overload_capacity_base = float(overload_instance.routes.loc[overload_mask, "capacity_units"].iloc[0])
    overload_capacity_new = overload_capacity_base * 0.80
    overload_instance.routes.loc[overload_mask, "capacity_units"] = overload_capacity_new
    overload = solve_instance(overload_instance, method="highs-ds", presolve=True)
    if not overload.result.success:
        raise RuntimeError(f"Arc-overload scenario failed unexpectedly: {overload.result.message}")
    _save_solution("overload", overload)
    save_solver_log(overload, RESULTS / "solver_overload.log", scipy.__version__, highs_version, presolve=True)
    verification_overload = verify_from_csv(
        overload_instance,
        RESULTS / "overload_solution.csv",
        expected_objective=float(overload.result.fun),
    )
    save_verification(verification_overload, RESULTS / "verification_overload.json")
    if not verification_overload["feasible"]:
        raise RuntimeError("Independent verification failed for arc-overload scenario")

    overload_flow = overload.solution[["warehouse_id", "store_id", "flow_units"]].rename(columns={"flow_units": "overload_flow"})
    overload_changes = base_flow.merge(overload_flow, on=["warehouse_id", "store_id"], how="outer").fillna(0.0)
    overload_changes["delta_flow"] = overload_changes["overload_flow"] - overload_changes["base_flow"]
    overload_changes = overload_changes[overload_changes["delta_flow"].abs() > 1e-7].sort_values(
        "delta_flow", key=np.abs, ascending=False
    )
    overload_changes.to_csv(RESULTS / "overload_flow_changes.csv", index=False, float_format="%.9f")

    growth_instance = _copy_instance(instance)
    growth_instance.stores["demand_units"] *= 1.20
    growth = solve_instance(growth_instance, method="highs-ds", presolve=True)
    save_solver_log(growth, RESULTS / "solver_demand_growth.log", scipy.__version__, highs_version, presolve=True)
    growth_info = {
        "status": status_name(growth.result),
        "message": str(growth.result.message),
        "total_supply_units": float(growth_instance.warehouses["supply_units"].sum()),
        "total_demand_units": float(growth_instance.stores["demand_units"].sum()),
        "aggregate_shortage_units": float(
            growth_instance.stores["demand_units"].sum() - growth_instance.warehouses["supply_units"].sum()
        ),
    }
    (RESULTS / "demand_growth_diagnosis.json").write_text(
        json.dumps(growth_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    greedy_solution, greedy_summary = greedy_cheapest_routes(instance)
    greedy_solution.to_csv(RESULTS / "greedy_solution.csv", index=False, float_format="%.9f")
    greedy_verification = verify_from_csv(instance, RESULTS / "greedy_solution.csv", expected_objective=greedy_summary["objective_cost"])
    save_verification(greedy_verification, RESULTS / "verification_greedy.json")
    if not greedy_summary["feasible"] or not greedy_verification["feasible"]:
        raise RuntimeError("Greedy heuristic is not feasible for the generated instance")
    greedy_cost = float(greedy_summary["objective_cost"])
    optimal_cost = float(base.result.fun)
    greedy_gap = 100.0 * (greedy_cost - optimal_cost) / optimal_cost
    greedy_summary["optimal_cost"] = optimal_cost
    greedy_summary["absolute_gap"] = greedy_cost - optimal_cost
    greedy_summary["gap_percent"] = greedy_gap
    (RESULTS / "greedy_summary.json").write_text(
        json.dumps(greedy_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    milp = solve_integer_instance(instance, presolve=True)
    if not milp.result.success:
        raise RuntimeError(f"MILP extension failed: {milp.result.message}")
    milp.solution.to_csv(RESULTS / "milp_solution.csv", index=False, float_format="%.9f")
    save_milp_log(milp, RESULTS / "solver_milp.log", scipy.__version__, highs_version, presolve=True)
    verification_milp = verify_integer_from_csv(
        instance, RESULTS / "milp_solution.csv", expected_objective=float(milp.result.fun)
    )
    save_verification(verification_milp, RESULTS / "verification_milp.json")
    if not verification_milp["feasible"]:
        raise RuntimeError("Independent verification failed for MILP extension")

    lp_milp = base.solution[["warehouse_id", "store_id", "flow_units"]].rename(columns={"flow_units": "lp_flow"})
    lp_milp = lp_milp.merge(
        milp.solution[["warehouse_id", "store_id", "flow_units"]].rename(columns={"flow_units": "milp_flow"}),
        on=["warehouse_id", "store_id"],
        how="outer",
    ).fillna(0.0)
    lp_milp["delta_flow"] = lp_milp["milp_flow"] - lp_milp["lp_flow"]
    lp_milp = lp_milp[lp_milp["delta_flow"].abs() > 1e-7].sort_values(
        "delta_flow", key=np.abs, ascending=False
    )
    lp_milp.to_csv(RESULTS / "lp_milp_flow_changes.csv", index=False, float_format="%.9f")

    milp_cost = float(milp.result.fun)
    integrality_gap = 100.0 * (milp_cost - optimal_cost) / optimal_cost
    milp_summary = {
        "status": "optimal",
        "lp_objective_cost": optimal_cost,
        "milp_objective_cost": milp_cost,
        "absolute_gap": milp_cost - optimal_cost,
        "integrality_gap_percent": integrality_gap,
        "mip_node_count": int(getattr(milp.result, "mip_node_count", 0) or 0),
        "mip_gap": float(getattr(milp.result, "mip_gap", 0.0) or 0.0),
        "elapsed_s": float(milp.elapsed_s),
        "max_integrality_violation": verification_milp["max_integrality_violation"],
        "changed_routes": int(len(lp_milp)),
    }
    (RESULTS / "milp_summary.json").write_text(
        json.dumps(milp_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    scenarios = pd.DataFrame([
        {
            "scenario": "base",
            "status": "optimal",
            "objective_cost": optimal_cost,
            "delta_vs_base": 0.0,
            "change": "исходная сеть",
            "iterations": int(base.result.nit),
        },
        {
            "scenario": "arc_failure",
            "status": "optimal",
            "objective_cost": float(failure.result.fun),
            "delta_vs_base": float(failure.result.fun) - optimal_cost,
            "change": f"закрыт маршрут {failed_route[0]}→{failed_route[1]}",
            "iterations": int(failure.result.nit),
        },
        {
            "scenario": "arc_overload",
            "status": "optimal",
            "objective_cost": float(overload.result.fun),
            "delta_vs_base": float(overload.result.fun) - optimal_cost,
            "change": f"пропускная способность {overload_route[0]}→{overload_route[1]} -20%",
            "iterations": int(overload.result.nit),
        },
        {
            "scenario": "demand_plus_20",
            "status": status_name(growth.result),
            "objective_cost": np.nan,
            "delta_vs_base": np.nan,
            "change": "спрос всех магазинов +20%",
            "iterations": int(getattr(growth.result, "nit", 0) or 0),
        },
    ])
    scenarios.to_csv(RESULTS / "scenarios.csv", index=False, float_format="%.9f")

    plot_network(instance, base.solution, PLOTS / "network_base.png", "Базовый оптимальный поток")
    plot_network(failure_instance, failure.solution, PLOTS / "network_failure.png", "Перестроение потока после отказа маршрута", failed_route=failed_route)
    plot_network(overload_instance, overload.solution, PLOTS / "network_overload.png", "Перестроение потока при перегрузке маршрута")
    plot_scenario_costs(scenarios, PLOTS / "scenario_costs.png")
    plot_bottlenecks(base.bottlenecks, PLOTS / "bottleneck_shadow_prices.png")
    plot_optimal_vs_greedy(optimal_cost, greedy_cost, PLOTS / "optimal_vs_greedy.png")
    plot_lp_vs_milp(optimal_cost, milp_cost, PLOTS / "lp_vs_milp.png")

    run_info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "highs": highs_version,
        "variant": VARIANT,
        "seed": SEED,
        "base_objective_cost": optimal_cost,
        "base_iterations": int(base.result.nit),
        "independent_verification": verification_base,
        "most_loaded_route": f"{failed_route[0]}->{failed_route[1]}",
        "most_loaded_route_flow": float(failed_row["flow_units"]),
        "arc_failure_objective_cost": float(failure.result.fun),
        "arc_failure_damage": float(failure.result.fun) - optimal_cost,
        "overload_route": f"{overload_route[0]}->{overload_route[1]}",
        "overload_capacity_base": overload_capacity_base,
        "overload_capacity_new": overload_capacity_new,
        "overload_objective_cost": float(overload.result.fun),
        "overload_damage": float(overload.result.fun) - optimal_cost,
        "overload_shadow_prediction": float(critical_bottleneck["capacity_shadow_benefit"]) * (overload_capacity_base - overload_capacity_new),
        "demand_growth_status": status_name(growth.result),
        "demand_growth_shortage_units": growth_info["aggregate_shortage_units"],
        "greedy_cost": greedy_cost,
        "greedy_gap_percent": greedy_gap,
        "milp_objective_cost": milp_cost,
        "milp_integrality_gap_percent": integrality_gap,
    }
    (RESULTS / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Variant: {VARIANT}; seed: {SEED}")
    print(f"Base: {status_name(base.result)}, cost={optimal_cost:.3f}")
    print(f"Arc failure {failed_route[0]}->{failed_route[1]}: cost={float(failure.result.fun):.3f}")
    print(f"Arc overload {overload_route[0]}->{overload_route[1]} -20% capacity: cost={float(overload.result.fun):.3f}")
    print(f"Demand +20%: {status_name(growth.result)}, shortage={growth_info['aggregate_shortage_units']:.1f}")
    print(f"Greedy: cost={greedy_cost:.3f}, gap={greedy_gap:.3f}%")
    print(f"Bottlenecks: {len(base.bottlenecks)}")
    print(f"MILP: cost={milp_cost:.3f}, gap={integrality_gap:.3f}%")


if __name__ == "__main__":
    main()
