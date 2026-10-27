"""Unit tests for the axis-profile metric maths, on synthetic vectors with known answers.

The anchors (noop, oracle) in every results table already exercise two of these
cases, but they cannot catch a sign error or a missing sqrt: a bug that made
`profile_error` reward under-editing would still put noop at 1 and oracle at 0.
These tests pin the whole curve.

    python3 scripts/bench_v5/test_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_profile  # noqa: E402


def make(src, tgt, out):
    return score_profile.metrics(np.array(src, dtype=float), np.array(tgt, dtype=float),
                                 np.array(out, dtype=float),
                                 np.array([1.0, 0.0]), np.array([1.0, 0.0]),
                                 np.array([1.0, 0.0]))


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def main() -> int:
    rng = np.random.default_rng(0)
    tp = rng.normal(size=6)
    src = rng.normal(size=6)
    tgt = src + tp
    fails = []

    def check(name, got, want):
        if not close(got, want, 1e-9):
            fails.append(f"{name}: got {got:.6f}, want {want:.6f}")

    # 1. noop: output == source
    m = make(src, tgt, src)
    check("noop.delivery", m["delivery"], 0.0)
    check("noop.profile_error", m["profile_error"], 1.0)
    check("noop.waste", m["waste"], 0.0)
    check("noop.magnitude", m["magnitude"], 0.0)

    # 2. oracle: output == target
    m = make(src, tgt, tgt)
    check("oracle.delivery", m["delivery"], 1.0)
    check("oracle.profile_error", m["profile_error"], 0.0)
    check("oracle.waste", m["waste"], 0.0)
    check("oracle.magnitude", m["magnitude"], 1.0)
    check("oracle.alignment", m["alignment"], 1.0)

    # 3. scaled along the target direction: delivery = a, error = |a - 1|
    for a in (0.25, 0.5, 0.9, 1.1, 2.0, 3.0):
        m = make(src, tgt, src + a * tp)
        check(f"a={a}.delivery", m["delivery"], a)
        check(f"a={a}.profile_error", m["profile_error"], abs(a - 1.0))
        check(f"a={a}.waste", m["waste"], 0.0)

    # 4. purely orthogonal edit: no delivery, all of it waste
    orth = rng.normal(size=6)
    orth -= orth @ tp / (tp @ tp) * tp
    orth *= np.linalg.norm(tp) / np.linalg.norm(orth)
    m = make(src, tgt, src + orth)
    check("orthogonal.delivery", m["delivery"], 0.0)
    check("orthogonal.waste", m["waste"], 1.0)
    # An edit that is entirely off-target is *worse than doing nothing*: it is
    # |tp| short along the target's direction and |tp| away from it sideways, so
    # the error is sqrt(2), not 1. This is the property that stops an arm from
    # winning by editing hard in the wrong direction.
    check("orthogonal.profile_error", m["profile_error"], float(np.sqrt(2.0)))

    # 5. error is the hypotenuse of (delivery - 1) and waste
    v = src + 0.7 * tp + 0.4 * orth
    m = make(src, tgt, v)
    check("combined.profile_error", m["profile_error"],
          float(np.hypot(m["delivery"] - 1.0, m["waste"])))

    # 6. a degenerate target (||tp|| = 0) must be refused, not divided by zero
    if score_profile.metrics(src, src, src, np.array([1.0, 0.0]),
                             np.array([1.0, 0.0]), np.array([1.0, 0.0])) is not None:
        fails.append("degenerate target: expected None, got a result")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        return 1
    print("ok: 25 assertions (noop, oracle, 6 scales, orthogonal, hypotenuse, degenerate)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
