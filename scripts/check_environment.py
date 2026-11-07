"""Read-only check of the existing runtime environment (Python >= 3.8).

No packages are installed, upgraded or downgraded. Reports use the configured
path. --check-imports checks binary imports and a small CPU model forward.
--gpu also checks small stage-one and stage-two forward/backward computations.
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
sys.path.insert(0, str(ROOT))

from model_probe import model_training_probe


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


def inspect_packages(rules, reference):
    rows = []
    for key, (name, operator, minimum, rule) in rules.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        supported = satisfies(installed, operator, minimum)
        reference_rule = reference.get(key)
        rows.append({"package": name, "installed": installed, "requirement": rule,
                     "status": "ok" if supported else ("missing" if installed is None else "incompatible"),
                     "reference_requirement": reference_rule[3] if reference_rule else None,
                     "reference_matches": satisfies(installed, reference_rule[1], reference_rule[2]) if reference_rule else None})
    return rows


def check_imports(seed=0):
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
        import torch
        torch.set_num_threads(1)
        result["cpu_model_and_numpy_bridge"] = {"ok": True, **model_training_probe("cpu", seed)}
    except Exception as exc:
        result["cpu_model_and_numpy_bridge"] = {"ok": False, "error": type(exc).__name__ + ": " + str(exc)}
    return result


def check_gpu(cfg, physical_gpu, seed=42):
    try:
        import torch
        from asei.pipeline import get_device
        from asei.resources import configure_cpu
        configure_cpu(cfg)
        device = get_device(cfg["runtime"]["device"])
        torch.cuda.reset_peak_memory_stats(device)
        probe = model_training_probe(device, seed, cfg["stage2"])
        torch.cuda.synchronize(device)
        return {"ok": True, "physical_gpu": physical_gpu, "logical_device": str(device), "seed": seed,
                "name": torch.cuda.get_device_name(device), "cuda": torch.version.cuda,
                "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device), **probe}
    except Exception as exc:
        return {"ok": False, "physical_gpu": physical_gpu, "seed": seed,
                "error": type(exc).__name__ + ": " + str(exc)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", "--config-path", default=str(ROOT / "configs" / "psm.yaml"))
    parser.add_argument("--strict-numpy", "--strict-tab", action="store_true", help="Also require the exact numpy==1.21.0 runtime pin")
    parser.add_argument("--check-imports", action="store_true")
    parser.add_argument("--gpu", "--gpus", type=int, help="Check one physical GPU; also enables import checks")
    parser.add_argument("--seed", type=int, default=42, help="Seed for synthetic model probes (default: 42)")
    parser.add_argument("--tools", action="store_true", help="Require optional download/test utilities too")
    parser.add_argument("--output", type=Path, help="Override reports.environment; no environment changes")
    args = parser.parse_args()
    reference = requirements(ROOT / "configs" / "environment_reference.txt")
    rules = requirements(ROOT / "requirements.txt")
    if args.strict_numpy:
        rules["numpy"] = reference["numpy"]
    core = inspect_packages(rules, reference)
    tools_file = ROOT / "requirements-tools.txt"
    tool_rules = (requirements(tools_file) if tools_file.is_file() else {
        "gdown": ("gdown", ">=", "5.2", "gdown>=5.2"),
        "pytest": ("pytest", ">=", "7.0", "pytest>=7.0"),
    })
    optional = inspect_packages(tool_rules, reference)
    python_ok = sys.version_info >= (3, 8)
    ok = python_ok and all(row["status"] == "ok" for row in core)
    if args.tools:
        ok = ok and all(row["status"] == "ok" for row in optional)
    cfg, gpu = None, None
    if args.gpu is not None and ok:
        try:
            sys.path.insert(0, str(ROOT))
            from asei.config import load_config, configure_cuda
            from asei.resources import configure_cpu
            cfg = load_config(args.config)
            configure_cuda(cfg, gpu=args.gpu)
            configure_cpu(cfg)  # Set visibility and thread limits before importing numeric libraries.
        except Exception as exc:
            gpu = {"ok": False, "physical_gpu": args.gpu, "seed": args.seed,
                   "error": type(exc).__name__ + ": " + str(exc)}
    imports = check_imports(args.seed) if (args.check_imports or args.gpu is not None) and ok else {}
    ok = ok and all(row["ok"] for row in imports.values())
    if args.gpu is not None:
        gpu = gpu or (check_gpu(cfg, args.gpu, args.seed) if ok else {
            "ok": False, "physical_gpu": args.gpu, "seed": args.seed,
            "error": "Dependency/import checks must pass before the GPU probe."})
        ok = ok and gpu["ok"]
    report = {"python": platform.python_version(), "executable": sys.executable,
              "python_supported": python_ok, "strict_numpy": args.strict_numpy,
              "core": core, "optional_tools": optional, "imports": imports, "gpu": gpu, "ok": ok}
    print("Python:", report["python"], "Executable:", sys.executable)
    print("{:<18} {:<18} {:<24} {}".format("Package", "Installed", "ASEI requirement", "Status"))
    for row in core:
        print("{:<18} {:<18} {:<24} {}".format(row["package"], row["installed"] or "missing",
                                               row["requirement"], row["status"]))
    print("Optional tools:", ", ".join(row["package"] + "=" + (row["installed"] or "missing") for row in optional))
    numpy = next(row for row in core if canonical_name(row["package"]) == "numpy")
    if numpy["reference_requirement"] and not numpy["reference_matches"]:
        print("The reference profile pins numpy==1.21.0; the default profile accepts your newer installed NumPy.")
    for name, row in imports.items():
        print("Import/probe:", name, "ok" if row["ok"] else row["error"])
    if gpu is not None:
        print("GPU probe:", f"physical={args.gpu} logical={gpu.get('logical_device')} {gpu.get('name')} seed={args.seed} ok"
              if gpu["ok"] else gpu["error"])
    target = ((args.output if args.output.is_absolute() else ROOT / args.output).resolve()
              if args.output else None)
    try:
        sys.path.insert(0, str(ROOT))
        from asei.config import load_config, project_path, report_path
        target = project_path(target) if target else report_path(load_config(args.config), "environment")
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
        print("PyYAML is missing; the configured report path cannot be read until it is installed.")
    if target:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("Environment report:", target)
    print("Environment check:", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
