import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.report_sspa_epsilon_trajectories import load_suite


class ReportTests(unittest.TestCase):
    def write_fixture(self, root):
        config = dict(seed=9001, policy="fixed_common", batch_size=2, samples=4,
                      calibration_anchors=8, validation_anchors=4, validation_samples=8)
        counts = dict(training_anchors=200, training_oracle_outputs=800,
                      training_generated_outputs=800, training_reference_outputs=800,
                      calibration_oracle_outputs=64, validation_oracle_outputs=128,
                      diagnostic_oracle_outputs=32)
        manifest = dict(seeds=[9001], policies=["fixed_common"], updates=100,
                        checkpoints=[0, 100], configs=[config])
        folder = root / "fixed_common/seed9001"
        folder.mkdir(parents=True)
        (root / "manifest.json").write_text(json.dumps(manifest))
        result = dict(config=config, final_update=100, counts=counts,
                      selected_update=100, selected_anchor_swd=.1, last_anchor_swd=.1)
        history = [dict(update=0, anchor_swd=1., counts={}), dict(update=100, anchor_swd=.1, counts=counts)]
        for name, data in (("result.json", result), ("trajectory.json", history),
                           ("training_trace.json", [dict(update=100)]),
                           ("status.json", dict(state="complete", update=100))):
            (folder / name).write_text(json.dumps(data))
        return folder

    def test_valid_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = self.write_fixture(root)
            self.assertEqual(len(load_suite(root)[1]), 1)
            (folder / "result.json").unlink()
            with self.assertRaisesRegex(ValueError, "Missing"):
                load_suite(root)

    def test_bad_counts_and_selection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = self.write_fixture(root)
            path = folder / "result.json"
            original = json.loads(path.read_text())
            result = copy.deepcopy(original)
            result["counts"]["training_oracle_outputs"] = 100
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "count"):
                load_suite(root)
            result = copy.deepcopy(original)
            result["selected_update"] = 0
            path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "Selection"):
                load_suite(root)

    def test_duplicate_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            folder = self.write_fixture(root)
            path = folder / "trajectory.json"
            history = json.loads(path.read_text())
            history.append(history[-1])
            path.write_text(json.dumps(history))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_suite(root)
