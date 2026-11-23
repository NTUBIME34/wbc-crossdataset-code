#!/usr/bin/env python3
"""Plan, preflight and execute the study's real commands without claiming untested work."""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
VARIABLE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")
KINDS = {"evidence", "inference", "statistics", "figure", "preparation", "training", "fit", "manual"}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def replace_axes(value, axes):
    if isinstance(value, str):
        for key, item in axes.items():
            value = value.replace("{" + key + "}", str(item))
        return value
    if isinstance(value, list):
        return [replace_axes(v, axes) for v in value]
    if isinstance(value, dict):
        return {k: replace_axes(v, axes) for k, v in value.items()}
    return value


def expand_axes(entry):
    axes = entry.get("foreach", {})
    keys = list(axes)
    for values in itertools.product(*(axes[k] for k in keys)):
        selected = dict(zip(keys, values))
        for key, value in list(selected.items()):
            if isinstance(axes[key], dict):
                selected.update(axes[key][value])
        copy = replace_axes({k: v for k, v in entry.items() if k != "foreach"}, selected)
        yield copy


def load_registry(path):
    registry = json.loads(Path(path).read_text())
    if registry.get("schema_version") != 1:
        raise ValueError("Unsupported registry schema")
    tasks = [task for family in registry["tasks"] for task in expand_axes(family)]
    ids = [t["id"] for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate expanded task ID")
    for task in tasks:
        if task.get("kind") not in KINDS or not task.get("paper_items"):
            raise ValueError(f"Invalid kind or missing paper coverage: {task['id']}")
        if task["kind"] != "manual" and not task.get("argv"):
            raise ValueError(f"No executable command: {task['id']}")
    return registry, tasks


def expand(value, env, strict=True):
    missing = []
    for _ in range(20):
        def sub(match):
            key = match.group(1)
            if key not in env or not env[key]:
                missing.append(key)
                return match.group(0)
            return str(env[key])
        new = VARIABLE.sub(sub, str(value))
        if new == value:
            break
        value = new
    if strict and (missing or VARIABLE.search(str(value))):
        raise ValueError("Unset or cyclic variable(s): " + ", ".join(sorted(set(missing or VARIABLE.findall(str(value))))))
    return str(value)


def outside_repo(path, repo=REPO):
    path = Path(path).expanduser().resolve()
    repo = Path(repo).resolve()
    if path == repo or repo in path.parents:
        raise ValueError("Output must be outside the repository")
    return path


def task_environment(registry, task, config, parent_env=None):
    env = dict(os.environ if parent_env is None else parent_env)
    variables = config.get("variables", {})
    if any(k in variables for k in ("HOME", "home", "REPO")):
        raise ValueError("Configuration must not replace HOME, home or REPO")
    env.update({k: str(v) for k, v in variables.items()})
    env["REPO"] = str(REPO)
    for k, v in registry.get("defaults", {}).items():
        env.setdefault(k, str(v))
    interpreter = config.get("interpreters", {}).get(task.get("environment", "journal_env"), env.get("PYTHON", sys.executable))
    env["PYTHON"] = str(interpreter)
    env.update({k: expand(v, env, strict=False) for k, v in task.get("environment_overrides", {}).items()})
    for _ in range(20):
        before = dict(env)
        for k in registry.get("defaults", {}):
            env[k] = expand(env[k], env, strict=False)
        if env == before:
            break
    # Re-evaluate task-specific values after defaults have resolved.
    env.update({k: expand(v, env, strict=False) for k, v in task.get("environment_overrides", {}).items()})
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def paths_for(requirement, env):
    patterns = requirement.get("any_of", [requirement.get("path", "")])
    patterns = [expand(p, env) for p in patterns]
    kind = requirement.get("kind", "file")
    expected = requirement.get("minimum", 1)
    alternatives = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern, recursive=True)) if kind == "glob" else [pattern]
        if kind in ("file", "glob"):
            matches = [p for p in matches if Path(p).is_file() and Path(p).stat().st_size > 0]
        elif kind == "directory":
            matches = [p for p in matches if Path(p).is_dir()]
        else:
            raise ValueError(f"Unknown input kind: {kind}")
        alternatives.append(dict(pattern=pattern, matches=matches))
        if len(matches) >= expected:
            return True, alternatives
    return False, alternatives


def inspect_task(registry, task, config, check_outputs=False):
    result = {"id": task["id"], "kind": task["kind"], "paper_items": task["paper_items"],
              "environment": task.get("environment"), "limitations": task.get("limitations", []),
              "checks": [], "numerical_agreement": "NOT_ASSESSED"}
    if task["kind"] == "manual":
        result.update(state="MANUAL", reason=task["reason"])
        return result, None
    try:
        env = task_environment(registry, task, config)
        if "OUT_ROOT" not in env or not env["OUT_ROOT"]:
            raise ValueError("Set OUT_ROOT explicitly outside the repository")
        env["OUT_ROOT"] = str(outside_repo(expand(env["OUT_ROOT"], env)))
        argv = [expand(v, env) for v in task["argv"]]
        executable = argv[0]
        if not shutil.which(executable):
            raise ValueError(f"Executable not available: {executable}")
        # Shell runners invoke PYTHON, so check that interpreter as well.
        if not shutil.which(expand(env["PYTHON"], env)):
            raise ValueError(f"Environment interpreter not available: {env['PYTHON']}")
        result["argv"] = argv
        result["interpreter"] = env["PYTHON"]
        fields = "outputs" if check_outputs else "inputs"
        for family in task.get(fields, []):
            for requirement in expand_axes(family):
                ok, alternatives = paths_for(requirement, env)
                result["checks"].append(dict(passed=ok, description=requirement.get("description", fields),
                                              alternatives=alternatives))
        result["state"] = "READY" if all(c["passed"] for c in result["checks"]) else ("MISSING_OUTPUTS" if check_outputs else "MISSING_INPUTS")
        return result, env
    except (ValueError, OSError) as exc:
        result.update(state="CONFIGURATION_ERROR", reason=str(exc))
        return result, None


def execute(argv, env, log_path, timeout):
    started = time.monotonic()
    with Path(log_path).open("x") as stream:
        proc = subprocess.Popen(argv, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
            return {"exit_code": rc, "elapsed_seconds": time.monotonic() - started, "timed_out": False}
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            return {"exit_code": proc.returncode, "elapsed_seconds": time.monotonic() - started, "timed_out": True}


def output_snapshot(task, env):
    found = {}
    for family in task.get("outputs", []):
        for requirement in expand_axes(family):
            _, alternatives = paths_for(requirement, env)
            for alternative in alternatives:
                for name in alternative["matches"]:
                    stat = Path(name).stat()
                    found[name] = [stat.st_size, stat.st_mtime_ns]
    return found


def main(argv=None):
    os.umask(0o077)
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="List all tasks; no experiment execution (default)")
    mode.add_argument("--check", action="store_true", help="Preflight selected tasks; never runs models")
    mode.add_argument("--run", action="store_true", help="Execute selected real commands, with logs outside the repo")
    ap.add_argument("--tasks", default="all", help="Comma-separated expanded task IDs, or all")
    ap.add_argument("--config", type=Path, help="JSON with variables and per-environment interpreters")
    ap.add_argument("--registry", type=Path, default=Path(__file__).with_name("study_tasks.json"))
    ap.add_argument("--allow-training", action="store_true", help="Allow selected network training tasks (may take days)")
    ap.add_argument("--timeout-seconds", type=float, default=86400, help="Per-task timeout; terminate its process group")
    ap.add_argument("--keep-going", action="store_true", help="Continue other tasks after failure; final exit remains nonzero")
    args = ap.parse_args(argv)
    try:
        if args.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        registry, all_tasks = load_registry(args.registry)
        requested = args.tasks.split(",")
        ids = {t["id"] for t in all_tasks}
        if args.tasks != "all" and (len(set(requested)) != len(requested) or set(requested) - ids):
            raise ValueError("Unknown or duplicate task IDs: " + args.tasks)
        tasks = all_tasks if args.tasks == "all" else [next(t for t in all_tasks if t["id"] == key) for key in requested]
        if not args.check and not args.run:
            for task in tasks:
                print(f"{task['id']}\t{task['kind']}\t{', '.join(task['paper_items'])}")
            print(f"{len(tasks)} tasks. Planning is not experimental verification.")
            return 0
        config = json.loads(args.config.read_text()) if args.config else {}
        if not isinstance(config, dict) or not isinstance(config.get("variables", {}), dict) or not isinstance(config.get("interpreters", {}), dict):
            raise ValueError("Configuration must contain object-valued variables/interpreters")
        out = outside_repo(config.get("variables", {}).get("OUT_ROOT", os.environ.get("OUT_ROOT", "")))
        if not config.get("variables", {}).get("OUT_ROOT", os.environ.get("OUT_ROOT", "")):
            raise ValueError("Set OUT_ROOT explicitly outside the repository")
        run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8]
        run_dir = out / "reproduction_runs" / run_id
        run_dir.mkdir(parents=True)
        shutil.copyfile(args.registry, run_dir / "study_tasks.snapshot.json")
        shutil.copyfile(Path(__file__), run_dir / "run_study.snapshot.py")
        sources = {}
        for folder in ("pipeline", "data_prep", "eval", "stats", "train", "figures", "reproduce", "environment"):
            for path in (REPO / folder).rglob("*"):
                if path.is_file() and path.suffix in {".py", ".sh", ".json", ".yml"}:
                    sources[path.relative_to(REPO).as_posix()] = sha(path)
        report = {"schema_version": 1, "mode": "run" if args.run else "check", "run_id": run_id,
                  "registry_sha256": sha(args.registry), "runner_sha256": sha(Path(__file__)),
                  "source_sha256": sources,
                  "scope": "Command/input/output audit, NOT a claim of numerical replication or all-study reproduction.",
                  "selected_task_ids": [t["id"] for t in tasks],
                  "results": []}
        failed = False
        for task in tasks:
            result, env = inspect_task(registry, task, config)
            if args.run and task["kind"] == "training" and not args.allow_training:
                result.update(state="TRAINING_NOT_AUTHORIZED", reason="Use --allow-training explicitly")
            elif args.run and result["state"] == "READY":
                log = run_dir / (task["id"] + ".log")
                before = output_snapshot(task, env)
                result["execution"] = execute(result["argv"], env, log, args.timeout_seconds)
                result["log"] = str(log)
                result["log_sha256"] = sha(log)
                if result["execution"]["timed_out"]:
                    result["state"] = "TIMED_OUT"
                elif result["execution"]["exit_code"] != 0:
                    result["state"] = "COMMAND_FAILED"
                else:
                    outputs, _ = inspect_task(registry, task, config, check_outputs=True)
                    result["output_checks"] = outputs["checks"]
                    result["state"] = "EXECUTED_OUTPUTS_PRESENT" if outputs["state"] == "READY" else "MISSING_OUTPUTS"
                    after = output_snapshot(task, env)
                    result["output_freshness"] = {p: before.get(p) != stat for p, stat in after.items()}
                    result["output_sha256"] = {p: sha(p) for p in after}
                    if task["kind"] != "evidence" and (not after or not all(result["output_freshness"].values())):
                        result["state"] = "UNCHANGED_OR_UNASSERTED_OUTPUTS"
                    result["numerical_agreement"] = "NOT_ASSESSED"
                    if task["kind"] == "evidence":
                        result["state"] = "EVIDENCE_CHECK_EXECUTED"
            report["results"].append(result)
            print(f"{result['id']}: {result['state']}", flush=True)
            ok = result["state"] in {"READY", "EXECUTED_OUTPUTS_PRESENT", "EVIDENCE_CHECK_EXECUTED"}
            failed |= not ok
            report["all_selected_tasks_executable_or_executed"] = not failed
            report["complete_study_reproduced"] = False
            report["not_attempted"] = [t["id"] for t in tasks[len(report["results"]):]]
            (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            if not ok and not args.keep_going:
                break
        print(f"Report: {run_dir / 'report.json'}")
        return 1 if failed else 0
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
