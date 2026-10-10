from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from conditional_drifting.sspa_trajectory import SSPATrajectory, TrajectoryConfig, source_hashes
from scripts.run_sspa_epsilon_trajectories import (continuation_plan, initialize_continuation,
                                                  file_hash, write_json)
from scripts.audit_sspa_continuation import audit


class ContinuationTests(unittest.TestCase):
    device = os.environ.get("TRAJECTORY_TEST_DEVICE", "cpu")

    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def parent(self, root):
        cfg = TrajectoryConfig(batch_size=8, hidden_dim=16, calibration_anchors=8,
                               validation_anchors=4, validation_samples=8, validation_projections=4)
        a = SSPATrajectory(cfg, self.device)
        folder = root / cfg.policy / f"seed{cfg.seed}"
        folder.mkdir(parents=True)
        a.validate()
        a.save(folder / "update0.pt")
        for i in range(100):
            a.step(diagnostics=i == 99)
        a.validate()
        a.save(folder / "update100.pt")
        selected = min(a.history, key=lambda row: (row["anchor_swd"], row["update"]))
        write_json(folder / "result.json", dict(config=asdict(cfg), final_update=100, counts=a.counts,
            selected_update=selected["update"], selected_anchor_swd=selected["anchor_swd"],
            last_anchor_swd=a.history[-1]["anchor_swd"], fixed_epsilon=a.fixed_epsilon,
            train_seconds=a.train_seconds, validation_seconds=a.validation_seconds))
        write_json(folder / "trajectory.json", a.history)
        write_json(folder / "training_trace.json", a.trace)
        write_json(folder / "status.json", dict(state="complete", update=100))
        manifest = dict(seeds=[cfg.seed], policies=[cfg.policy], configs=[asdict(cfg)],
            updates=100, checkpoints=[0, 100], source_hashes=source_hashes(),
            device=self.device, torch_version=str(torch.__version__),
            gpu=torch.cuda.get_device_name() if self.device == "cuda" else None)
        write_json(root / "manifest.json", manifest)
        return a, folder, manifest | dict(updates=200, checkpoints=[0, 100, 200])

    def test_continuation_exact_and_parent_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "parent"
            a, parent_folder, manifest = self.parent(root)
            plan = continuation_plan(root, manifest)
            hashes_before = {p.name: file_hash(p) for p in parent_folder.iterdir()}
            for _ in range(2):
                a.step(True)
            b = SSPATrajectory(a.cfg, self.device)
            folder = Path(d) / "child"
            folder.mkdir()
            initialize_continuation(folder, a.cfg.policy, a.cfg.seed, plan, b)
            for _ in range(2):
                b.step(True)
            for key, tensor in a.model.state_dict().items():
                self.assertTrue(torch.equal(tensor, b.model.state_dict()[key]))
            self.assertEqual(a.counts, b.counts)
            self.assertEqual(a.trace, b.trace)
            self.assertEqual(hashes_before, {p.name: file_hash(p) for p in parent_folder.iterdir()})

    def test_reject_budget_source_and_schedule_changes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _, _, manifest = self.parent(root)
            for replacement, message in ((dict(updates=100), "increase"),
                    (dict(checkpoints=[0, 50, 100, 200]), "earlier"),
                    (dict(source_hashes={}), "source_hashes")):
                with self.assertRaisesRegex(ValueError, message):
                    continuation_plan(root, manifest | replacement)

    def test_reject_checkpoint_record_mismatch_and_tampering(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "parent"
            a, folder, manifest = self.parent(root)
            plan = continuation_plan(root, manifest)
            path = folder / "update100.pt"
            state = torch.load(path, weights_only=False, map_location="cpu")
            state["update"] = 99
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "checkpoint/record"):
                continuation_plan(root, manifest)
            child = Path(d) / "child"
            child.mkdir()
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                initialize_continuation(child, a.cfg.policy, a.cfg.seed, plan, a)

    def test_complete_audit_and_inherited_corruption(self):
        with tempfile.TemporaryDirectory() as d:
            parent, child = Path(d) / "parent", Path(d) / "child"
            a, _, manifest = self.parent(parent)
            plan = continuation_plan(parent, manifest)
            folder = child / a.cfg.policy / f"seed{a.cfg.seed}"
            folder.mkdir(parents=True)
            initialize_continuation(folder, a.cfg.policy, a.cfg.seed, plan, a)
            for i in range(100):
                a.step(diagnostics=i == 99)
            a.validate()
            a.save(folder / "update200.pt")
            selected = min(a.history, key=lambda row: (row["anchor_swd"], row["update"]))
            write_json(child / "manifest.json", manifest | dict(continuation=plan))
            write_json(folder / "result.json", dict(config=asdict(a.cfg), final_update=200, counts=a.counts,
                selected_update=selected["update"], selected_anchor_swd=selected["anchor_swd"],
                last_anchor_swd=a.history[-1]["anchor_swd"], fixed_epsilon=a.fixed_epsilon,
                train_seconds=a.train_seconds, validation_seconds=a.validation_seconds))
            write_json(folder / "trajectory.json", a.history)
            write_json(folder / "training_trace.json", a.trace)
            write_json(folder / "status.json", dict(state="complete", update=200))
            result = audit(child, parent)
            self.assertEqual(result["inherited_checkpoints_verified"], 2)
            self.assertEqual(result["tasks"][0]["added_counts"]["training_oracle_outputs"], 3200)
            path = folder / "update0.pt"
            path.write_bytes(path.read_bytes() + b"tamper")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                audit(child, parent)


if __name__ == "__main__":
    unittest.main()
