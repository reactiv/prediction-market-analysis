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


def backtest(name: str | None = None, force: bool = False):
    """Run a backtest strategy by name or show interactive menu."""
    from dataclasses import replace

    from src.backtest.engine import BacktestRunner
    from src.backtest.strategies.calibration_surface import calibration_surface
    from src.backtest.strategies.fade_longshot import fade_longshot

    runner = BacktestRunner()

    strategies = {
        "fade_longshot": fade_longshot(),
        "fade_longshot_5c": fade_longshot(max_price=5),
        "fade_longshot_10c": fade_longshot(max_price=10),
        "fade_longshot_20c": fade_longshot(max_price=20),
        "fade_longshot_no_fees": replace(
            fade_longshot(fee_rate=0.0), name="fade_longshot_lt15c_no_fees"
        ),
        "cal_surface": calibration_surface(),
        "cal_surface_15c": calibration_surface(max_price=15),
        "cal_surface_5pp": calibration_surface(min_mae_7d=0.05),
        "cal_surface_10pp": calibration_surface(min_mae_7d=0.10),
    }

    if name:
        if name == "all":
            print("\nRunning all backtests...\n")
            for strat in strategies.values():
                runner.run(strat, force=force)
            print("\nAll backtests complete.")
            return

        if name in strategies:
            runner.run(strategies[name], force=force)
            return

        print(f"Strategy '{name}' not found. Available strategies:")
        for key, strat in strategies.items():
            print(f"  - {key}: {strat.description}")
        sys.exit(1)

    # Interactive menu
    options = ["[All] Run all backtests"]
    strat_keys = list(strategies.keys())
    for key in strat_keys:
        strat = strategies[key]
        options.append(f"{key}: {strat.description}")
    options.append("[Exit]")

    menu = TerminalMenu(
        options,
        title="Select a backtest to run (use arrow keys):",
        cycle_cursor=True,
        clear_screen=False,
    )
    choice = menu.show()

    if choice is None or choice == len(options) - 1:
        print("Exiting.")
        return

    if choice == 0:
        backtest("all", force=force)
    else:
        key = strat_keys[choice - 1]
        runner.run(strategies[key], force=force)


def simulate(name: str | None = None):
    """Run Monte Carlo simulation + Kelly sizing on a completed backtest."""
    from src.backtest.distributions import analyze_distribution
    from src.backtest.kelly import compute_kelly
    from src.backtest.monte_carlo import run_monte_carlo

    base = Path("data") / "backtests"

    if not name:
        # List available backtests
        if not base.exists():
            print("No backtests found. Run 'backtest' first.")
            return
        dirs = sorted(d.name for d in base.iterdir() if (d / "returns.parquet").exists())
        if not dirs:
            print("No completed backtests found.")
            return

        options = list(dirs) + ["[Exit]"]
        menu = TerminalMenu(
            options,
            title="Select a backtest to simulate:",
            cycle_cursor=True,
            clear_screen=False,
        )
        choice = menu.show()
        if choice is None or choice == len(options) - 1:
            print("Exiting.")
            return
        name = dirs[choice]

    backtest_dir = base / name
    if not (backtest_dir / "returns.parquet").exists():
        print(f"No returns.parquet in {backtest_dir}")
        sys.exit(1)

    print(f"\n=== Simulation: {name} ===\n")

    # Step 1: Distribution analysis
    print("1. Analyzing return distribution...")
    dist = analyze_distribution(backtest_dir)
    print(f"   n={dist['n']:,}, mean={dist['mean']:.4f}c, std={dist['std']:.4f}c")
    print(f"   skew={dist['skewness']:.4f}, kurtosis={dist['kurtosis']:.4f}")
    print(f"   Normal: {dist['normality']['is_normal']}")
    print(f"   Student-t df={dist['student_t_fit']['df']:.2f}")

    # Step 2: Monte Carlo
    print("\n2. Running Monte Carlo (10K paths)...")
    mc = run_monte_carlo(backtest_dir)
    print(f"   {mc['elapsed_seconds']:.1f}s, {mc['trades_per_path']:,} trades/path")
    print(f"   PnL median: {mc['pnl']['median']:,.0f}c")
    print(f"   PnL 5th-95th: [{mc['pnl']['p5']:,.0f}, {mc['pnl']['p95']:,.0f}]")
    dd = mc["max_drawdown"]
    print(f"   Drawdown p50: {dd['median']:,.0f}c, p95: {dd['p95']:,.0f}c, p99: {dd['p99']:,.0f}c")

    # Step 3: Kelly sizing
    print("\n3. Computing Kelly position sizes...")
    kelly = compute_kelly(backtest_dir, mc_summary=mc)
    print(f"   Win prob: {kelly['win_probability']:.4f}")
    print(f"   Payoff ratio: {kelly['payoff_ratio']}")
    print(f"   Kelly fraction: {kelly['kelly_fraction']:.4f}")
    print(f"   CV of edge: {kelly['cv_edge']:.4f}")
    print(f"   Adjusted Kelly: {kelly['kelly_adjusted']:.4f}")
    print(f"   Half-Kelly: {kelly['half_kelly']:.4f}")
    print(f"   Avg cost/contract: {kelly['avg_cost_per_contract']:.2f}c")

    print(f"\n   Output: {backtest_dir}/")


def sweep(force: bool = False):
    """Sweep calibration surface thresholds to find optimal entry rules."""
    from src.backtest.engine import BacktestRunner
    from src.backtest.strategies.calibration_surface import calibration_sweep

    runner = BacktestRunner()
    strategies = calibration_sweep()

    print(f"\nSweeping {len(strategies)} thresholds...\n")
    print(f"{'Threshold':>10} {'Trades':>10} {'Win Rate':>10} {'Avg Ret':>10} {'Sharpe':>10}")
    print("-" * 55)

    for strat in strategies:
        output_dir = runner.run(strat, force=force)
        manifest = output_dir / "manifest.json"
        if manifest.exists():
            import json

            data = json.loads(manifest.read_text())
            print(
                f"{strat.name:>10} "
                f"{data['qualifying_trades']:>10,} "
                f"{data['win_rate']:>10.4f} "
                f"{data['avg_return_cents']:>10.4f} "
                f"{data['sharpe_ratio']:>10.4f}"
            )

    print("\nSweep complete. Results in data/backtests/")


def package():
    """Package the data directory into a zstd-compressed tar archive."""
    success = package_data()
    sys.exit(0 if success else 1)


def main():
    if len(sys.argv) < 2:
        print("\nUsage: uv run main.py <command>")
        print("Commands: analyze, index, transform, backtest, simulate, sweep, package")
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

    if command == "backtest":
        force = "--force" in sys.argv
        args = [a for a in sys.argv[2:] if a != "--force"]
        name = args[0] if args else None
        backtest(name, force=force)
        sys.exit(0)

    if command == "simulate":
        name = sys.argv[2] if len(sys.argv) > 2 else None
        simulate(name)
        sys.exit(0)

    if command == "sweep":
        force = "--force" in sys.argv
        sweep(force=force)
        sys.exit(0)

    if command == "package":
        package()
        sys.exit(0)

    print(f"Unknown command: {command}")
    print("Commands: analyze, index, transform, backtest, simulate, sweep, package")
    sys.exit(1)


if __name__ == "__main__":
    main()
