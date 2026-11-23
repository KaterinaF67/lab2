from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

N = 25
Q = 1
Y = 26
VARIANT = 1 + ((17 * N + 7 * Q + Y) % 20)
SEED = 10000 + 100 * Q + N
N_WAREHOUSES = 4
N_STORES = 6
N_FORBIDDEN = 5


@dataclass
class NetworkInstance:
    warehouses: pd.DataFrame
    stores: pd.DataFrame
    routes: pd.DataFrame


def _candidate(rng: np.random.Generator) -> NetworkInstance:
    supply = rng.integers(45, 71, N_WAREHOUSES)
    total = int(supply.sum())

    raw = rng.uniform(0.8, 1.2, N_STORES)
    demand = np.floor(raw / raw.sum() * total).astype(int)
    remainder = total - int(demand.sum())
    for k in range(remainder):
        demand[k % N_STORES] += 1

    wx = rng.uniform(0, 20, N_WAREHOUSES)
    wy = rng.uniform(0, 20, N_WAREHOUSES)
    sx = rng.uniform(0, 20, N_STORES)
    sy = rng.uniform(0, 20, N_STORES)

    dist = np.sqrt((wx[:, None] - sx[None, :]) ** 2 + (wy[:, None] - sy[None, :]) ** 2)
    cost = np.round(2 + 0.9 * dist + rng.uniform(0, 2, (N_WAREHOUSES, N_STORES)), 2)

    allowed = np.ones((N_WAREHOUSES, N_STORES), dtype=bool)
    ids = list(range(N_WAREHOUSES * N_STORES))
    rng.shuffle(ids)
    forbidden = []
    for z in ids:
        i, j = divmod(z, N_STORES)
        if len(forbidden) >= N_FORBIDDEN:
            break
        if allowed[i].sum() <= 3 or allowed[:, j].sum() <= 3:
            continue
        allowed[i, j] = False
        forbidden.append((i, j))

    capacity = np.zeros((N_WAREHOUSES, N_STORES), dtype=float)
    for i in range(N_WAREHOUSES):
        for j in range(N_STORES):
            if allowed[i, j]:
                base = min(supply[i], demand[j])
                capacity[i, j] = max(12.0, round(base * rng.uniform(0.45, 0.95), 1))

    for i in range(N_WAREHOUSES):
        need = float(supply[i] - capacity[i].sum())
        if need > 0:
            js = np.where(allowed[i])[0]
            capacity[i, rng.choice(js)] += need + 5.0

    for j in range(N_STORES):
        need = float(demand[j] - capacity[:, j].sum())
        if need > 0:
            ii = np.where(allowed[:, j])[0]
            capacity[rng.choice(ii), j] += need + 5.0

    warehouses = pd.DataFrame({
        "warehouse_id": [f"W{i+1}" for i in range(N_WAREHOUSES)],
        "supply_units": supply.astype(float),
        "x": np.round(wx, 6),
        "y": np.round(wy, 6),
    })
    stores = pd.DataFrame({
        "store_id": [f"M{j+1}" for j in range(N_STORES)],
        "demand_units": demand.astype(float),
        "x": np.round(sx, 6),
        "y": np.round(sy, 6),
    })

    rows = []
    for i in range(N_WAREHOUSES):
        for j in range(N_STORES):
            rows.append({
                "warehouse_id": f"W{i+1}",
                "store_id": f"M{j+1}",
                "allowed": bool(allowed[i, j]),
                "unit_cost": float(cost[i, j]),
                "capacity_units": float(capacity[i, j]) if allowed[i, j] else 0.0,
            })
    routes = pd.DataFrame(rows)
    return NetworkInstance(warehouses, stores, routes)


def generate_instance(seed: int = SEED) -> NetworkInstance:
    rng = np.random.default_rng(seed)
    instance = None
    for _ in range(VARIANT):
        instance = _candidate(rng)
    return instance


def save_instance(instance: NetworkInstance, data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    instance.warehouses.to_csv(data_dir / "warehouses.csv", index=False, float_format="%.6f")
    instance.stores.to_csv(data_dir / "stores.csv", index=False, float_format="%.6f")
    instance.routes.to_csv(data_dir / "routes.csv", index=False, float_format="%.6f")
    metadata = {
        "student_number_N": N,
        "group_index_Q": Q,
        "year_Y": Y,
        "variant_formula": "1 + ((17*N + 7*Q + Y) mod 20)",
        "variant": VARIANT,
        "seed_formula": "10000 + 100*Q + N",
        "seed": SEED,
        "warehouses": N_WAREHOUSES,
        "stores": N_STORES,
        "forbidden_routes": N_FORBIDDEN,
        "generation": "The RNG with the fixed seed generates V consecutive candidate networks; variant V uses the V-th candidate.",
    }
    (data_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def load_instance(data_dir: Path) -> NetworkInstance:
    warehouses = pd.read_csv(data_dir / "warehouses.csv")
    stores = pd.read_csv(data_dir / "stores.csv")
    routes = pd.read_csv(data_dir / "routes.csv")
    routes["allowed"] = routes["allowed"].astype(bool)
    return NetworkInstance(warehouses, stores, routes)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    save_instance(generate_instance(), root / "data")
