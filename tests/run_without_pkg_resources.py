"""Run a real CLI with a legacy-import gate, including spawned workers.

Set DSRX_MIGRATION_ENTRYPOINT to binarize or train, then pass the normal CLI
arguments to this file. This is validation instrumentation, not a runtime shim.
"""

import importlib.abc
import os
from pathlib import Path
import runpy
import sys

if "pkg_resources" in sys.modules:
    raise AssertionError("pkg_resources was loaded before the import gate")


class RejectPkgResources(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pkg_resources" or fullname.startswith("pkg_resources."):
            raise AssertionError(f"Forbidden legacy import: {fullname}")


sys.meta_path.insert(0, RejectPkgResources())
entrypoint = os.environ["DSRX_MIGRATION_ENTRYPOINT"]
if entrypoint not in ("binarize", "train"):
    raise ValueError("DSRX_MIGRATION_ENTRYPOINT must be binarize or train")
repository = Path(__file__).resolve().parents[1]
namespace = runpy.run_path(
    str(repository / "scripts" / f"{entrypoint}.py"), run_name="__dependency_probe__"
)
if __name__ == "__main__":
    namespace["binarize" if entrypoint == "binarize" else "run_task"]()
    if "pkg_resources" in sys.modules:
        raise AssertionError("pkg_resources was loaded during validation")
    print("PASS: entrypoint completed without pkg_resources", flush=True)
