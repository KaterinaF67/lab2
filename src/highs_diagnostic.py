from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_instance
from .model import solve_instance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--method", default="highs-ds")
    args = parser.parse_args()

    instance = load_instance(args.data)
    output = solve_instance(instance, method=args.method, presolve=True, disp=True)
    result = output.result
    print("\n--- SciPy/HiGHS result summary ---")
    print(f"success={result.success}")
    print(f"status={result.status}")
    print(f"message={result.message}")
    print(f"objective={getattr(result, 'fun', None)}")
    print(f"iterations={getattr(result, 'nit', None)}")
    print(f"crossover_iterations={getattr(result, 'crossover_nit', None)}")
    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
