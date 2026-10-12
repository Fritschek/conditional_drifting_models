import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

spec = importlib.util.spec_from_file_location("intervention_report", Path(__file__).resolve().parents[1] / "scripts/report_sspa_interventions.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class InterventionReportTests(unittest.TestCase):
    def test_incomplete_suite_not_aggregated(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                report.report(Path(directory))

    def test_late_gate_uses_all_requested_checkpoints(self):
        parent = dict(anchor_swd=.04, anchor_swd_floor=.02, anchor_mean_l2=.03, anchor_mean_l2_floor=.02)
        rows = [dict(update=u, anchor_swd=.05, conditional_variance=.05, anchor_mean_l2=.04)
                for u in (300000, 350000, 390720)]
        self.assertTrue(report.bounded(rows, parent, .32))
        self.assertEqual(rows[0]["update"], 300000)
        self.assertFalse(report.bounded(rows[:2], parent, .32))
        rows[1]["anchor_swd"] = .2
        self.assertFalse(report.bounded(rows, parent, .32))
