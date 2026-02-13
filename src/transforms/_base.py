"""Base class for data transforms that derive new datasets from raw data.

Usage:
    from src.transforms._base import Transform

    class MyTransform(Transform):
        def run(self) -> None:
            # Process data and write outputs
            pass

    transform = MyTransform("t1a", "Trade-level enrichment")
    transform.run()
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from tqdm import tqdm


class Transform(ABC):
    """Base class for data transforms.

    Subclasses implement `run()` to process raw data and write derived datasets.
    """

    def __init__(self, name: str, description: str, dependencies: list[str] | None = None):
        self.name = name
        self.description = description
        self.dependencies = dependencies or []
        self.base_dir = Path(__file__).parent.parent.parent
        self.output_dir = self.base_dir / "data" / "transforms" / self.name

    @contextmanager
    def progress(self, description: str) -> Generator[None, None, None]:
        """Show a progress spinner while executing a block of code."""
        with tqdm(
            total=None,
            desc=description,
            bar_format="{desc}: {elapsed}",
            file=sys.stderr,
            leave=False,
        ) as pbar:
            yield
            pbar.update()

    def check_dependencies(self) -> bool:
        """Verify all dependency manifest.json files exist."""
        for dep in self.dependencies:
            manifest = self.base_dir / "data" / "transforms" / dep / "manifest.json"
            if not manifest.exists():
                print(f"Missing dependency: {dep} (no manifest.json at {manifest})")
                return False
        return True

    def ensure_output_dir(self) -> Path:
        """Create and return the output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def write_manifest(self, metadata: dict) -> None:
        """Write manifest.json with run metadata."""
        manifest = {
            "transform": self.name,
            "description": self.description,
            "dependencies": self.dependencies,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **metadata,
        }
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
        print(f"Wrote manifest: {manifest_path}")

    @abstractmethod
    def run(self) -> None:
        """Execute the transform."""
        pass

    def execute(self, force: bool = False) -> None:
        """Run the transform with dependency checking and timing."""
        manifest_path = self.output_dir / "manifest.json"
        if not force and manifest_path.exists():
            print(f"Skipping {self.name}: already completed (use --force to re-run)")
            return

        if not self.check_dependencies():
            print(f"Cannot run {self.name}: missing dependencies")
            return

        print(f"Running transform: {self.name} - {self.description}")
        start = time.time()
        self.run()
        elapsed = time.time() - start
        print(f"Transform {self.name} completed in {elapsed:.1f}s")

    @classmethod
    def load(cls, transform_dir: Path | str = "src/transforms") -> list[type[Transform]]:
        """Scan directory for Transform subclass implementations."""
        transform_dir = Path(transform_dir)
        if not transform_dir.exists():
            return []

        transforms: list[type[Transform]] = []

        for py_file in transform_dir.glob("**/*.py"):
            if py_file.name.startswith("_"):
                continue

            relative_path = py_file.relative_to(transform_dir)
            module_parts = relative_path.with_suffix("").parts
            module_name = "src.transforms." + ".".join(module_parts)
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue

            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, cls) and obj is not cls and not inspect.isabstract(obj):
                    transforms.append(obj)

        return transforms
