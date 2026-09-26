from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--results-dir",
        default="cross_layer/results",
    )
    p.add_argument("--seed", type=int, default=20170922)
    p.add_argument("--updates", type=int, default=10000)
    args = p.parse_args()

    rows = []

    for group in ("g2", "g3", "g4"):
        path = (
            Path(args.results_dir)
            / f"{group}_seed{args.seed}_u{args.updates}"
            / "summary.json"
        )

        if not path.exists():
            rows.append((group, None))
            continue

        rows.append((group, json.loads(path.read_text())))

    print()
    print("Cross-layer pilot summary")
    print("=" * 78)
    print(
        f"{'Group':<8}"
        f"{'Best Val':>12}"
        f"{'Best Step':>12}"
        f"{'Corr Mix':>12}"
        f"{'Corr Cut':>12}"
        f"{'Minutes':>12}"
    )
    print("-" * 78)

    for group, data in rows:
        if data is None:
            print(f"{group:<8}{'MISSING':>12}")
            continue

        def fmt(v):
            return "-" if v is None else f"{v:+.3f}"

        print(
            f"{group:<8}"
            f"{data['best_val_acc1']:>11.2f}%"
            f"{data['best_val_step']:>12d}"
            f"{fmt(data.get('mean_corr_mix')):>12}"
            f"{fmt(data.get('mean_corr_cut')):>12}"
            f"{data['elapsed_seconds']/60:>12.1f}"
        )

    present = {
        g: d for g, d in rows if d is not None
    }

    if "g3" in present and "g4" in present:
        delta = (
            present["g4"]["best_val_acc1"]
            - present["g3"]["best_val_acc1"]
        )
        print()
        print(
            f"CORE PILOT: G4 - G3 = {delta:+.2f} pp "
            f"(validation, single seed)"
        )
        print(
            "Interpret only as a pilot signal; full training + multiple seeds "
            "are required for a research claim."
        )


if __name__ == "__main__":
    main()
