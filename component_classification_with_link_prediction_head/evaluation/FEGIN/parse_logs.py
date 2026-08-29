from collections import defaultdict
import os
import re
from pathlib import Path


def get_long_path(path_obj: Path) -> Path:
    abs_str = str(path_obj.resolve())
    if os.name == "nt" and not abs_str.startswith("\\\\?\\"):
        return Path("\\\\?\\" + abs_str)
    return path_obj


def parse_fegin_logs():
    base_dir = get_long_path(Path.cwd())

    # SORTED LONGEST FIRST so 'component_pin_net' matches before 'component_pin'
    target_granularities = [
        "component_pin_net",
        "component_component",
        "component_pin",
        "component_net",
    ]

    times_data = {
        "demos": defaultdict(list),
        "examples": defaultdict(list),
    }

    time_regex = re.compile(
        r"Training completed in\s+([\d.]+)\s+seconds", re.IGNORECASE
    )

    if not base_dir.exists():
        print(f"Error: Base directory '{base_dir}' does not exist.")
        return

    try:
        search_dirs = list(base_dir.iterdir())
    except Exception as e:
        print(f"Error listing base directory: {e}")
        return

    valid_roots = 0
    for search_dir in search_dirs:
        if not search_dir.is_dir() or search_dir.name.endswith("old"):
            continue

        dir_name = search_dir.name
        if dir_name.startswith("hyperparameter_search_ltspice_demos_"):
            dataset_type = "demos"
        elif dir_name.startswith("hyperparameter_search_ltspice_examples_"):
            dataset_type = "examples"
        else:
            continue

        valid_roots += 1

        try:
            subdirs = list(search_dir.iterdir())
        except Exception as e:
            print(f"Warning: Could not read {search_dir}: {e}")
            continue

        for run_dir in subdirs:
            if not run_dir.is_dir():
                continue

            run_name = run_dir.name

            detected_granularity = None
            for gran in target_granularities:
                if run_name.startswith(gran):
                    detected_granularity = gran
                    break

            if not detected_granularity:
                continue

            output_file = run_dir / "output.txt"
            if output_file.is_file():
                try:
                    content = output_file.read_text(
                        encoding="utf-8", errors="ignore"
                    )
                    match = time_regex.search(content)
                    if match:
                        training_time = float(match.group(1))
                        times_data[dataset_type][detected_granularity].append(
                            training_time
                        )
                except Exception as e:
                    print(f"Error reading {output_file}: {e}")

    print(f"Scanned {valid_roots} search root directories.\n")

    print("=" * 65)
    print("FEGIN TRAINING TIME SUMMARY REPORT")
    print("=" * 65)

    # Print results in original logical order
    display_order = [
        "component_component",
        "component_net",
        "component_pin",
        "component_pin_net",
    ]

    for ds_key, ds_label in [
        ("demos", "LTSpice Demos"),
        ("examples", "LTSpice Examples"),
    ]:
        print(f"\n--- {ds_label} ---")
        print(
            f"{'Granularity':<25} | {'Count':<8} | {'Avg Training Time (s)':<22}"
        )
        print("-" * 65)

        for gran in display_order:
            times = times_data[ds_key][gran]
            count = len(times)
            if count > 0:
                avg_time = sum(times) / count
                print(f"{gran:<25} | {count:<8} | {avg_time:<22.2f}")
            else:
                print(f"{gran:<25} | {0:<8} | N/A")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    parse_fegin_logs()