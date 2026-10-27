#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def polynomial_mmd_x100(x: np.ndarray, y: np.ndarray) -> float:
    xx = (x @ x.T / x.shape[1] + 1.0) ** 2
    yy = (y @ y.T / y.shape[1] + 1.0) ** 2
    xy = (x @ y.T / x.shape[1] + 1.0) ** 2
    return float((xx.mean() + yy.mean() - 2.0 * xy.mean()) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    methods = json.loads((args.run_root / "methods.json").read_text(encoding="utf-8"))
    features = args.run_root / "jedi" / "features"
    high = np.load(features / "high.npy")
    rows = []
    for item in methods:
        method = item["method"]
        path = features / f"{method}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        value = polynomial_mmd_x100(np.load(path), high)
        rows.append({"method": method, "jedi_mmd_x100": value})

    out = args.run_root / "jedi" / "results"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "jedi_val50.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ["method", "jedi_mmd_x100"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "jedi_val50.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print((out / "jedi_val50.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
