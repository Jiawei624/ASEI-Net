"""Read-only check for reusing an existing TAB environment (Python >= 3.8).

No packages are installed, upgraded or downgraded. --check-imports additionally
checks binary imports, the NumPy/PyTorch bridge and a small CPU model forward.
"""
from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def canonical_name(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def requirements(path):
    result = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            result.update(requirements(Path(path).parent / line[3:].strip()))
            continue
        match = re.fullmatch(r"([\w.-]+)(?:(>=|==)([\w.+-]+))?", line)
        if not match:
            raise ValueError("Unsupported requirement in environment check: " + line)
        name, operator, version = match.groups()
        result[canonical_name(name)] = (name, operator, version, line)
    return result


def satisfies(installed, operator, minimum):
    if installed is None:
        return False
    if operator is None:
        return True

    def release(version):
        public = version.split("+", 1)[0]
        match = re.fullmatch(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?(.*)", public)
        if not match:
            raise ValueError("Cannot compare package version: " + version)
        return tuple(int(v or 0) for v in match.groups()[:3]), match.group(4)

    current, suffix = release(installed)
    wanted, _ = release(minimum)
    if operator == "==":
        return current == wanted and not suffix
    if current != wanted:
        return current > wanted
    return not suffix or suffix.startswith(".post")


def inspect_packages(rules, upstream):
    rows = []
    for key, (name, operator, minimum, rule) in rules.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        supported = satisfies(installed, operator, minimum)
        tab_rule = upstream.get(key)
        rows.append({"package": name, "installed": installed, "requirement": rule,
                     "status": "ok" if supported else ("missing" if installed is None else "incompatible"),
                     "tab_requirement": tab_rule[3] if tab_rule else None,
                     "tab_matches": satisfies(installed, tab_rule[1], tab_rule[2]) if tab_rule else None})
    return rows


def check_imports():
    result = {}
    for name in ("numpy", "pandas", "torch", "sklearn", "scipy", "statsmodels.tsa.stattools", "yaml", "tqdm"):
        try:
            importlib.import_module(name)
            result[name] = {"ok": True}
        except Exception as exc:
            result[name] = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)}
    if not all(row["ok"] for row in result.values()):
        return result
    try:
        import numpy as np
        import torch
        sys.path.insert(0, str(ROOT))
        from asei.data import Record
        from asei.model import NormalRepresentation, DualTrajectoryDetector

        torch.set_num_threads(1)
        torch.manual_seed(0)
        z = np.random.default_rng(0).normal(size=(40, 2)).astype(np.float32)
        model = NormalRepresentation({"channels": 2, "window": 32, "d_model": 16, "heads": 4,
                                      "layers": 1, "dropout": 0.0,
                                      "marginal_prototypes": 4, "joint_prototypes": 4}).eval()
        with torch.no_grad():
            model.descriptor.fit(Record(z, np.ones_like(z, bool)))
            model.memory.cache_decoders(model.history.identity.weight)
            history = torch.from_numpy(z[:32][None])
            prediction = model.predict(history, torch.ones_like(history, dtype=torch.bool))
            chi = model.descriptor(torch.from_numpy(z[1:33][None]))
            observed = torch.ones(1, 2, dtype=torch.bool)
            mi, ji = model.memory.query(chi, observed, model.history.identity.weight)
            rm, rj, mv, jv = model.memory.references(mi, ji)
            detector = DualTrajectoryDetector(model.history.identity.weight).eval()
            output, _ = detector.step({"chi": chi, "prediction": prediction,
                "residual": torch.from_numpy(z[32][None, :, None]) - prediction,
                "marginal": rm, "joint": rj, "m_valid": mv, "j_valid": jv,
                "observed": observed, "prediction_ready": observed})
            if not output["valid"].all() or not torch.isfinite(output["score"]).all():
                raise RuntimeError("Non-finite or invalid model probe output.")
            np.testing.assert_array_equal(torch.from_numpy(z).numpy(), z)
        result["cpu_model_and_numpy_bridge"] = {"ok": True}
    except Exception as exc:
        result["cpu_model_and_numpy_bridge"] = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-tab", action="store_true", help="Also require TAB's exact numpy==1.21.0 runtime pin")
    parser.add_argument("--check-imports", action="store_true")
    parser.add_argument("--tools", action="store_true", help="Require optional download/test utilities too")
    parser.add_argument("--output", type=Path, help="Optional JSON report; no environment changes")
    args = parser.parse_args()
    source = ROOT / "third_party" / "TAB" / "requirements.txt"
    upstream = requirements(source) if source.is_file() else {}
    rules = requirements(ROOT / ("requirements-tab.txt" if args.strict_tab else "requirements.txt"))
    core = inspect_packages(rules, upstream)
    optional = inspect_packages(requirements(ROOT / "requirements-tools.txt"), upstream)
    python_ok = sys.version_info >= (3, 8)
    ok = python_ok and all(row["status"] == "ok" for row in core)
    if args.tools:
        ok = ok and all(row["status"] == "ok" for row in optional)
    imports = check_imports() if args.check_imports and ok else {}
    ok = ok and all(row["ok"] for row in imports.values())
    report = {"python": platform.python_version(), "executable": sys.executable,
              "python_supported": python_ok, "strict_tab_numpy": args.strict_tab,
              "core": core, "optional_tools": optional, "imports": imports, "ok": ok}
    print("Python:", report["python"], "Executable:", sys.executable)
    print("{:<18} {:<18} {:<24} {}".format("Package", "Installed", "ASEI requirement", "Status"))
    for row in core:
        print("{:<18} {:<18} {:<24} {}".format(row["package"], row["installed"] or "missing",
                                               row["requirement"], row["status"]))
    print("Optional tools:", ", ".join(row["package"] + "=" + (row["installed"] or "missing") for row in optional))
    numpy = next(row for row in core if canonical_name(row["package"]) == "numpy")
    if numpy["tab_requirement"] and not numpy["tab_matches"]:
        print("TAB pins numpy==1.21.0; the default reuse profile accepts your newer installed NumPy.")
    for name, row in imports.items():
        print("Import/probe:", name, "ok" if row["ok"] else row["error"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Environment check:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
