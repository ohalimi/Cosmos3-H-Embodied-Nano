"""Per-Python-worker launch prefix. Without the flag exec Python unchanged (no observer import)."""

import os
import sys


def main(argv):
    if argv[:1] != ["--internal-signals"]:
        os.execv(sys.executable, [sys.executable, *argv])
    if len(argv) < 4 or argv[2] != "--":
        raise SystemExit(
            "usage: python launch.py --internal-signals PRIVATE.json -- SCRIPT [ARGS]"
        )
    from pathlib import Path
    import runpy

    root = Path(__file__).resolve().parents[1]
    # Do not put the observer worktree ahead of the training adapter. Only the
    # observer's own package is imported from this path, then it is removed.
    from importlib.util import module_from_spec, spec_from_file_location

    if "openh_signals" not in sys.modules:
        init = root / "openh_signals" / "__init__.py"
        spec = spec_from_file_location(
            "openh_signals", init, submodule_search_locations=[str(init.parent)]
        )
        mod = module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    from openh_signals.runtime_launch import attach_worker

    if argv[3] == "-m":
        if len(argv) < 5:
            raise SystemExit("missing worker module")
        module = argv[4]
        # A production worker package can have initializer side effects: do
        # not import/find_spec the target before all preflight conditions pass.
        sys.path[0] = os.getcwd()
        with attach_worker(Path(argv[1]), "module:" + module, argv[5:]):
            sys.argv = [module, *argv[5:]]
            runpy.run_module(module, run_name="__main__", alter_sys=True)
    else:
        script = Path(argv[3]).resolve(strict=True)
        sys.path[0] = str(script.parent)
        with attach_worker(Path(argv[1]), script, argv[4:]):
            sys.argv = [str(script), *argv[4:]]
            runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
