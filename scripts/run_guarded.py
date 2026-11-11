"""Run training in a fresh process and retain native failures and exit signals.

Example: python scripts/run_guarded.py --config configs/smd.yaml --gpu 0 --seed 42 --log
Use --cuda-debug for synchronous CUDA error locations. Other options go to asei run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_report(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def cgroup_memory():
    """Best-effort evidence, never infer an OOM solely from SIGKILL."""
    try:
        entry = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                     if line.startswith("0::"))
        root = Path("/sys/fs/cgroup").resolve()
        directory = (root / entry.lstrip("/")).resolve()
        if root not in (directory, *directory.parents):
            return None
        values = dict(line.split() for line in (directory / "memory.events").read_text().splitlines())
        return {"events": {k: int(v) for k, v in values.items()},
                "limit": (directory / "memory.max").read_text().strip(),
                "current_bytes": int((directory / "memory.current").read_text())}
    except (OSError, StopIteration, ValueError):
        return None


def process_memory(pid):
    try:
        wanted = {"VmRSS", "VmHWM", "VmSize", "VmPeak"}
        return {key: value.strip() for key, value in
                (line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines() if ":" in line)
                if key in wanted}
    except OSError:
        return None


def supervise(command, directory, env=None, poll_seconds=5.):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if any((directory / name).exists() for name in ("console.log", "status.json", "exit.json")):
        raise FileExistsError("Diagnostic files already exist; choose a new --report-dir.")
    console = directory / "console.log"
    record = dict(command=list(command), cwd=str(ROOT), started_utc=datetime.now(timezone.utc).isoformat(),
                  console_log=str(console), cgroup_before=cgroup_memory())
    began = time.monotonic()
    print("Training diagnostics:", directory, flush=True)
    with console.open("w", encoding="utf-8", buffering=1) as stream:
        try:
            process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=stream, stderr=subprocess.STDOUT)
        except OSError as exc:
            record.update(state="launch_failed", error=str(exc))
            write_report(directory / "exit.json", record)
            raise
        record.update(pid=process.pid, last_process_memory=None)
        try:
            while True:
                memory = process_memory(process.pid)
                if memory is not None:
                    record["last_process_memory"] = memory
                state = dict(record, state="running", elapsed_seconds=time.monotonic() - began)
                write_report(directory / "status.json", state)
                try:
                    code = process.wait(timeout=poll_seconds)
                    break
                except subprocess.TimeoutExpired:
                    continue
        except (KeyboardInterrupt, SystemExit):
            process.terminate()
            try:
                code = process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
            record["interrupted"] = True
        signame = signal.Signals(-code).name if code < 0 and os.name == "posix" else None
        record.update(state="succeeded" if code == 0 else "failed", returncode=code, signal=signame,
                      elapsed_seconds=time.monotonic() - began, cgroup_after=cgroup_memory(),
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_report(directory / "exit.json", record)
        write_report(directory / "status.json", record)
    print(f"Training {record['state']}: returncode={code} signal={signame}; report={directory / 'exit.json'}", flush=True)
    return (128 - code) if code < 0 else code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-debug", action="store_true")
    parser.add_argument("--report-dir", type=Path)
    args, training_args = parser.parse_known_args()
    directory = args.report_dir or (ROOT / "outputs/diagnostics" /
        (datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + f"_{os.getpid()}"))
    if not directory.is_absolute():
        directory = ROOT / directory
    env = os.environ.copy()
    env.update(PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    if args.cuda_debug:
        env["CUDA_LAUNCH_BLOCKING"] = "1"
    command = [sys.executable, "-u", "-X", "faulthandler", "-m", "asei", "run", *training_args]
    return supervise(command, directory, env)


if __name__ == "__main__":
    raise SystemExit(main())
