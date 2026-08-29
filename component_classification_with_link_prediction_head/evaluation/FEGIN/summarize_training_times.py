from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_PREFIXES = {
    "ltspice_demos": "hyperparameter_search_ltspice_demos",
    "ltspice_examples": "hyperparameter_search_ltspice_examples",
}
GRANULARITIES = [
    "component_component",
    "component_net",
    "component_pin",
    "component_pin_net",
]
TIME_RE = re.compile(r"Training completed in\s+([0-9]+(?:\.[0-9]+)?)\s+seconds")
GRANULARITY_RE = re.compile(r"^(component_component|component_net|component_pin|component_pin_net)")


def iter_valid_dirs() -> list[Path]:
    valid_dirs: list[Path] = []
    for candidate in sorted(ROOT.iterdir()):
        if not candidate.is_dir():
            continue
        name = candidate.name
        if not name.startswith("hyperparameter_search_ltspice_"):
            continue
        if name.endswith("old"):
            continue
        valid_dirs.append(candidate)
    return valid_dirs


def collect_timings() -> dict[str, dict[str, list[float]]]:
    timings: dict[str, dict[str, list[float]]] = {
        dataset: {granularity: [] for granularity in GRANULARITIES}
        for dataset in DATASET_PREFIXES
    }
    counts: dict[str, dict[str, int]] = {
        dataset: {granularity: 0 for granularity in GRANULARITIES}
        for dataset in DATASET_PREFIXES
    }

    for directory in iter_valid_dirs():
        dataset = None
        for key, prefix in DATASET_PREFIXES.items():
            if directory.name.startswith(prefix):
                dataset = key
                break
        if dataset is None:
            continue

        for output_file in sorted(directory.rglob("output.txt")):
            text = output_file.read_text(encoding="utf-8", errors="ignore")
            match = TIME_RE.search(text)
            if not match:
                continue

            seconds = float(match.group(1))
            granularity_match = GRANULARITY_RE.match(output_file.parent.name)
            if granularity_match is None:
                continue

            granularity = granularity_match.group(1)
            timings[dataset][granularity].append(seconds)
            counts[dataset][granularity] += 1

    return timings, counts


def main() -> None:
    timings, counts = collect_timings()

    print("Training time summary by dataset and granularity")
    print("=" * 70)

    for dataset in DATASET_PREFIXES:
        print(f"\n{dataset}")
        for granularity in GRANULARITIES:
            values = timings[dataset][granularity]
            if not values:
                print(f"  {granularity}: no valid runs found")
                continue
            avg = sum(values) / len(values)
            print(f"  {granularity}: count={len(values)}, avg={avg:.2f} s, min={min(values):.2f} s, max={max(values):.2f} s")

    print("\nCounts by granularity and dataset:")
    for dataset in DATASET_PREFIXES:
        print(f"  {dataset}: {', '.join(f'{g}={counts[dataset][g]}' for g in GRANULARITIES)}")


if __name__ == "__main__":
    main()
