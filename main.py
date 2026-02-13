from __future__ import annotations

import sys
from pathlib import Path

from simple_term_menu import TerminalMenu

from src.common.analysis import Analysis
from src.common.indexer import Indexer
from src.common.util import package_data
from src.common.util.strings import snake_to_title
from src.transforms._base import Transform


def analyze(name: str | None = None):
    """Run analysis by name or show interactive menu."""
    analyses = Analysis.load()

    if not analyses:
        print("No analyses found in src/analysis/")
        return

    output_dir = Path("output")

    # If name provided, run that specific analysis
    if name:
        if name == "all":
            print("\nRunning all analyses...\n")
            for analysis_cls in analyses:
                instance = analysis_cls()
                print(f"Running: {instance.name}")
                saved = instance.save(output_dir, formats=["png", "pdf", "csv", "json", "gif"])
                for fmt, path in saved.items():
                    print(f"  {fmt}: {path}")
            print("\nAll analyses complete.")
            return

        # Find matching analysis
        for analysis_cls in analyses:
            instance = analysis_cls()
            if instance.name == name:
                print(f"\nRunning: {instance.name}\n")
                saved = instance.save(output_dir, formats=["png", "pdf", "csv", "json", "gif"])
                print("Saved files:")
                for fmt, path in saved.items():
                    print(f"  {fmt}: {path}")
                return

        # No match found
        print(f"Analysis '{name}' not found. Available analyses:")
        for analysis_cls in analyses:
            instance = analysis_cls()
            print(f"  - {instance.name}")
        sys.exit(1)

    # Interactive menu mode
    options = ["[All] Run all analyses"]
    for analysis_cls in analyses:
        instance = analysis_cls()
        options.append(f"{snake_to_title(instance.name)}: {instance.description}")
    options.append("[Exit]")

    menu = TerminalMenu(
        options,
        title="Select an analysis to run (use arrow keys):",
        cycle_cursor=True,
        clear_screen=False,
    )
    choice = menu.show()

    if choice is None or choice == len(options) - 1:
        print("Exiting.")
        return

    if choice == 0:
        # Run all analyses
        print("\nRunning all analyses...\n")
        for analysis_cls in analyses:
            instance = analysis_cls()
            print(f"Running: {instance.name}")
            saved = instance.save(output_dir, formats=["png", "pdf", "csv", "json", "gif"])
            for fmt, path in saved.items():
                print(f"  {fmt}: {path}")
        print("\nAll analyses complete.")
    else:
        # Run selected analysis
        analysis_cls = analyses[choice - 1]
        instance = analysis_cls()
        print(f"\nRunning: {instance.name}\n")
        saved = instance.save(output_dir, formats=["png", "pdf", "csv", "json", "gif"])
        print("Saved files:")
        for fmt, path in saved.items():
            print(f"  {fmt}: {path}")


def index():
    """Interactive indexer selection menu."""
    indexers = Indexer.load()

    if not indexers:
        print("No indexers found in src/indexers/")
        return

    # Build menu options
    options = []
    for indexer_cls in indexers:
        instance = indexer_cls()
        options.append(f"{snake_to_title(instance.name)}: {instance.description}")
    options.append("[Exit]")

    menu = TerminalMenu(
        options,
        title="Select an indexer to run (use arrow keys):",
        cycle_cursor=True,
        clear_screen=False,
    )
    choice = menu.show()

    if choice is None or choice == len(options) - 1:
        print("Exiting.")
        return

    indexer_cls = indexers[choice]
    instance = indexer_cls()
    print(f"\nRunning: {instance.name}\n")
    instance.run()
    print("\nIndexer complete.")


def transform(name: str | None = None, force: bool = False):
    """Run a data transform by name or show interactive menu."""
    transforms = Transform.load()

    if not transforms:
        print("No transforms found in src/transforms/")
        return

    if name:
        if name == "all":
            print("\nRunning all transforms (respecting dependencies)...\n")
            completed = set()
            remaining = list(transforms)
            while remaining:
                runnable = [
                    t for t in remaining if all(d in completed for d in t().dependencies)
                ]
                if not runnable:
                    print("Cannot proceed: unresolvable dependencies")
                    for t in remaining:
                        inst = t()
                        print(f"  {inst.name}: blocked by {inst.dependencies}")
                    sys.exit(1)
                for transform_cls in runnable:
                    instance = transform_cls()
                    instance.execute(force=force)
                    completed.add(instance.name)
                    remaining.remove(transform_cls)
            print("\nAll transforms complete.")
            return

        for transform_cls in transforms:
            instance = transform_cls()
            if instance.name == name:
                instance.execute(force=force)
                return

        print(f"Transform '{name}' not found. Available transforms:")
        for transform_cls in transforms:
            instance = transform_cls()
            deps = f" (depends on: {', '.join(instance.dependencies)})" if instance.dependencies else ""
            print(f"  - {instance.name}{deps}")
        sys.exit(1)

    # Interactive menu mode
    options = ["[All] Run all transforms"]
    for transform_cls in transforms:
        instance = transform_cls()
        deps = f" [deps: {', '.join(instance.dependencies)}]" if instance.dependencies else ""
        options.append(f"{snake_to_title(instance.name)}: {instance.description}{deps}")
    options.append("[Exit]")

    menu = TerminalMenu(
        options,
        title="Select a transform to run (use arrow keys):",
        cycle_cursor=True,
        clear_screen=False,
    )
    choice = menu.show()

    if choice is None or choice == len(options) - 1:
        print("Exiting.")
        return

    if choice == 0:
        transform("all", force=force)
    else:
        transform_cls = transforms[choice - 1]
        instance = transform_cls()
        instance.execute(force=force)


def package():
    """Package the data directory into a zstd-compressed tar archive."""
    success = package_data()
    sys.exit(0 if success else 1)


def main():
    if len(sys.argv) < 2:
        print("\nUsage: uv run main.py <command>")
        print("Commands: analyze, index, transform, package")
        sys.exit(0)

    command = sys.argv[1]

    if command == "analyze":
        name = sys.argv[2] if len(sys.argv) > 2 else None
        analyze(name)
        sys.exit(0)

    if command == "index":
        index()
        sys.exit(0)

    if command == "transform":
        force = "--force" in sys.argv
        args = [a for a in sys.argv[2:] if a != "--force"]
        name = args[0] if args else None
        transform(name, force=force)
        sys.exit(0)

    if command == "package":
        package()
        sys.exit(0)

    print(f"Unknown command: {command}")
    print("Commands: analyze, index, transform, package")
    sys.exit(1)


if __name__ == "__main__":
    main()
