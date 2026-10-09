import copy
import unittest

from scripts.run_learned_metric_gate import CHEAP, METHODS, contrasts, refinement_decisions
from scripts.report_learned_metric_gate import validate


class LearnedMetricReportTests(unittest.TestCase):
    def setUp(self):
        rows, losses = [], []
        for a in range(3):
            for i, method in enumerate(METHODS):
                for rep in range(8):
                    key = {"samples": 512, "anchor": a, "method": method, "repeat": rep}
                    rows.append(key | {k: float(i) for k in CHEAP+("rbf_cross_trace",)})
                    losses.extend(key | {"probe": f"p{p}", "cross_squared_error": float(i)} for p in range(8))
        self.data = {"rows": rows, "losses": losses, "manifest": {
            "status": "complete", "config": {"preflight_only": False},
            "probes": [{"name": f"p{p}"} for p in range(8)],
            "refinement": refinement_decisions(contrasts(rows, losses, 512))}}

    def test_complete_panel_and_direction(self):
        checks = validate(self.data)
        self.assertEqual(checks["learned_metric_records"], 72)
        cs = contrasts(self.data["rows"], self.data["losses"], 512)
        self.assertEqual(len(cs), 72)
        self.assertTrue(all(r["kernel_agrees"] and not r["candidate_added_information"] for r in cs))
        self.assertTrue(all(r["target"]["ordering"] == -1 for r in cs))

    def test_missing_and_duplicate_rows(self):
        for key in ("rows", "losses"):
            data = copy.deepcopy(self.data)
            data[key].pop()
            with self.assertRaises(ValueError):
                validate(data)
            data = copy.deepcopy(self.data)
            data[key][-1] = data[key][0]
            with self.assertRaises(ValueError):
                validate(data)

    def test_reject_preflight_and_changed_refinement(self):
        data = copy.deepcopy(self.data)
        data["manifest"]["config"]["preflight_only"] = True
        with self.assertRaises(ValueError):
            validate(data)
        data = copy.deepcopy(self.data)
        data["manifest"]["refinement"][0]["unresolved_targets"] = 1
        with self.assertRaises(ValueError):
            validate(data)


if __name__ == "__main__":
    unittest.main()
