"""Offline tests of study coverage/preflight; no model or dataset download."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_study", ROOT / "reproduce/run_study.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="study_reproduction_test_")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.out = self.base / "outputs"
        self.config = {"variables": {"OUT_ROOT": str(self.out)}}

    def fixture_registry(self, task):
        path = self.base / "registry.json"
        path.write_text(json.dumps({"schema_version": 1, "tasks": [task]}))
        return path

    def task(self, **extra):
        task = dict(id="fixture", kind="statistics", paper_items=["fixture"],
                    environment="journal_env", argv=[sys.executable, "-c", "print('fixture')"])
        task.update(extra)
        return task

    def call_main(self, argv, task):
        config = self.base / "config.json"
        config.write_text(json.dumps(self.config))
        registry = self.fixture_registry(task)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = runner.main(["--registry", str(registry), "--config", str(config), *argv])
        reports = list(self.out.glob("reproduction_runs/*/report.json"))
        return result, json.loads(reports[-1].read_text()) if reports else None

    def test_registry_loads_and_has_unique_ids(self):
        _, tasks = runner.load_registry(ROOT / "reproduce/study_tasks.json")
        self.assertEqual(len(tasks), len({t["id"] for t in tasks}))
        self.assertGreater(len(tasks), 50)

    def test_registry_covers_every_main_table(self):
        _, tasks = runner.load_registry(ROOT / "reproduce/study_tasks.json")
        items = " ".join(i for t in tasks for i in t["paper_items"])
        for number in range(1, 5):   # main text has Tables 1-4 since the v2 restructure
            self.assertIn(f"Table {number}", items)

    def test_registry_covers_all_supplement_sections(self):
        _, tasks = runner.load_registry(ROOT / "reproduce/study_tasks.json")
        items = {item.split()[0] for t in tasks for item in t["paper_items"]}
        for number in range(1, 18):
            self.assertIn(f"S{number}", items)

    def test_no_duplicate_registry_id(self):
        path = self.base / "duplicate.json"
        path.write_text(json.dumps(dict(schema_version=1, tasks=[self.task(), self.task()])))
        with self.assertRaises(ValueError):
            runner.load_registry(path)

    def test_expand_axis_product(self):
        result = list(runner.expand_axes(dict(path="{model}-{seed}", foreach=dict(model=["a", "b"], seed=[1, 2]))))
        self.assertEqual([r["path"] for r in result], ["a-1", "a-2", "b-1", "b-2"])

    def test_named_axis_attributes(self):
        result = list(runner.expand_axes(dict(path="{arm}/{tag}", foreach=dict(arm={"a": {"tag": "one"}}))))
        self.assertEqual(result[0]["path"], "a/one")

    def test_expand_rejects_missing_variable(self):
        with self.assertRaises(ValueError):
            runner.expand("{MISSING}/x", {})

    def test_mixed_case_dataset_variable_expands(self):
        self.assertEqual(runner.expand("{LODO_TEST_ROOT_Raabin}/images", {"LODO_TEST_ROOT_Raabin": "dataset"}),
                         "dataset/images")

    def test_unset_mixed_case_dataset_variable_rejected(self):
        with self.assertRaises(ValueError):
            runner.expand("{LODO_TEST_ROOT_Raabin}/images", {})

    def test_expand_rejects_cycles(self):
        with self.assertRaises(ValueError):
            runner.expand("{A}", dict(A="{B}", B="{A}"))

    def test_output_inside_repo_rejected(self):
        with self.assertRaises(ValueError):
            runner.outside_repo(ROOT / "outputs")

    def test_output_symlink_into_repo_rejected(self):
        link = self.base / "link"
        link.symlink_to(ROOT, target_is_directory=True)
        with self.assertRaises(ValueError):
            runner.outside_repo(link / "outputs")

    def test_home_variable_override_rejected(self):
        with self.assertRaises(ValueError):
            runner.task_environment({}, self.task(), {"variables": {"HOME": "invalid"}})

    def test_missing_input_not_ready(self):
        result, _ = runner.inspect_task({}, self.task(inputs=[dict(path=str(self.base / "missing"))]), self.config)
        self.assertEqual(result["state"], "MISSING_INPUTS")

    def test_empty_file_not_ready(self):
        empty = self.base / "empty"
        empty.touch()
        result, _ = runner.inspect_task({}, self.task(inputs=[dict(path=str(empty))]), self.config)
        self.assertEqual(result["state"], "MISSING_INPUTS")

    def test_each_seed_is_required(self):
        (self.base / "seed_42").write_text("weight")
        task = self.task(inputs=[dict(path=str(self.base / "seed_{seed}"), foreach=dict(seed=[42, 62, 82]))])
        result, _ = runner.inspect_task({}, task, self.config)
        self.assertEqual(result["state"], "MISSING_INPUTS")
        self.assertEqual([c["passed"] for c in result["checks"]], [True, False, False])

    def test_input_alternative_can_satisfy(self):
        p = self.base / "fallback"
        p.write_text("weight")
        result, _ = runner.inspect_task({}, self.task(inputs=[dict(any_of=[str(self.base / "absent"), str(p)])]), self.config)
        self.assertEqual(result["state"], "READY")

    def test_glob_requires_minimum(self):
        (self.base / "one.csv").write_text("x")
        result, _ = runner.inspect_task({}, self.task(inputs=[dict(path=str(self.base / "*.csv"), kind="glob", minimum=2)]), self.config)
        self.assertEqual(result["state"], "MISSING_INPUTS")

    def test_unavailable_interpreter_not_ready(self):
        config = dict(self.config, interpreters={"journal_env": "/nonexistent/python"})
        result, _ = runner.inspect_task({}, self.task(), config)
        self.assertEqual(result["state"], "CONFIGURATION_ERROR")

    def test_manual_is_not_pass(self):
        rc, report = self.call_main(["--check"], self.task(kind="manual", reason="human input"))
        self.assertEqual(rc, 1)
        self.assertEqual(report["results"][0]["state"], "MANUAL")
        self.assertFalse(report["complete_study_reproduced"])

    def test_unknown_task_fails(self):
        rc, _ = self.call_main(["--check", "--tasks", "unknown"], self.task())
        self.assertEqual(rc, 2)

    def test_check_does_not_run_command(self):
        marker = self.base / "executed"
        task = self.task(argv=[sys.executable, "-c", f"open({str(marker)!r},'w').write('x')"])
        rc, report = self.call_main(["--check"], task)
        self.assertEqual(rc, 0)
        self.assertFalse(marker.exists())
        self.assertEqual(report["results"][0]["state"], "READY")

    def test_training_requires_opt_in(self):
        rc, report = self.call_main(["--run"], self.task(kind="training"))
        self.assertEqual(rc, 1)
        self.assertEqual(report["results"][0]["state"], "TRAINING_NOT_AUTHORIZED")

    def test_nonzero_command_fails(self):
        rc, report = self.call_main(["--run"], self.task(argv=[sys.executable, "-c", "raise SystemExit(7)"]))
        self.assertEqual(rc, 1)
        self.assertEqual(report["results"][0]["execution"]["exit_code"], 7)
        self.assertEqual(report["results"][0]["state"], "COMMAND_FAILED")

    def test_zero_exit_without_output_not_success(self):
        rc, report = self.call_main(["--run"], self.task(outputs=[dict(path=str(self.out / "result"))]))
        self.assertEqual(rc, 1)
        self.assertEqual(report["results"][0]["state"], "UNCHANGED_OR_UNASSERTED_OUTPUTS")

    def test_stale_output_not_success(self):
        self.out.mkdir()
        artifact = self.out / "result"
        artifact.write_text("old")
        rc, report = self.call_main(["--run"], self.task(outputs=[dict(path=str(artifact))]))
        self.assertEqual(rc, 1)
        self.assertEqual(report["results"][0]["state"], "UNCHANGED_OR_UNASSERTED_OUTPUTS")

    def test_new_output_is_command_execution_not_numerical_pass(self):
        artifact = self.out / "result"
        task = self.task(argv=[sys.executable, "-c", f"open({str(artifact)!r},'w').write('new')"], outputs=[dict(path=str(artifact))])
        rc, report = self.call_main(["--run"], task)
        self.assertEqual(rc, 0)
        self.assertEqual(report["results"][0]["state"], "EXECUTED_OUTPUTS_PRESENT")
        self.assertEqual(report["results"][0]["numerical_agreement"], "NOT_ASSESSED")
        self.assertFalse(report["complete_study_reproduced"])

    def test_timeout_fails(self):
        rc, report = self.call_main(["--run", "--timeout-seconds", "0.05"], self.task(argv=[sys.executable, "-c", "import time; time.sleep(10)"]))
        self.assertEqual(rc, 1)
        self.assertTrue(report["results"][0]["execution"]["timed_out"])
        self.assertEqual(report["results"][0]["state"], "TIMED_OUT")


if __name__ == "__main__":
    unittest.main()
