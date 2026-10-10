import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.run_sspa_epsilon_trajectories import main


class RunnerFailureTests(unittest.TestCase):
    def run_failure(self, root, keep_going, exception):
        created = []

        def trainer(cfg, device):
            created.append(cfg.policy)

            def validate():
                raise exception

            return SimpleNamespace(update=0, history=[], validate=validate,
                                   validation_seconds=0., train_seconds=0.)

        argv = ["runner", "--device", "cpu", "--out-dir", str(root), "--seeds", "9001",
                "--policies", "fixed_common,shared_adaptive", "--updates", "100"]
        if keep_going:
            argv.append("--keep-going")
        with patch("sys.argv", argv), patch("scripts.run_sspa_epsilon_trajectories.SSPATrajectory", trainer), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(type(exception) if not keep_going or isinstance(exception, KeyboardInterrupt) else RuntimeError):
                main()
        return created

    def test_records_both_failures_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertEqual(len(self.run_failure(root, True, ValueError("numerical failure"))), 2)
            failures = json.loads((root / "failed_tasks.json").read_text())
            self.assertEqual(len(failures), 2)
            for policy in ("fixed_common", "shared_adaptive"):
                folder = root / policy / "seed9001"
                self.assertEqual(json.loads((folder / "status.json").read_text())["state"], "failed")
                self.assertFalse((folder / "result.json").exists())

    def test_default_stops_after_first_failure(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(len(self.run_failure(Path(d), False, ValueError("failure"))), 1)

    def test_keyboard_interrupt_always_propagates(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(len(self.run_failure(Path(d), True, KeyboardInterrupt())), 1)
