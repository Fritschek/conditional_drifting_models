import copy
import unittest

from scripts.report_channel_metric_resolution import check_task_contract


class MetricReportContractTests(unittest.TestCase):
    def setUp(self):
        self.key = (7, "SSPA", "analytic")
        self.config = {
            "steps": [1e-4, 3e-4, 1e-3], "gradient_samples": 2048,
            "reference_samples": 16384, "eval_samples": 16384,
            "metric_samples": 1024, "repeats": 4,
        }
        self.payload = {
            "codec": {"sha256": "codec-a", "path": "original/codec.pt"},
            "noise_std": 0.3250531436, "rate": 0.75, "ebno_db": 8.0,
            "codec_config": {"message_dim": 64, "code_dim": 8,
                             "encoder_normalization": "standardize"},
            "methods": {"analytic": {"steps": [
                {"fraction": value} for value in self.config["steps"]
            ]}},
        }

    def register(self):
        contracts = {}
        check_task_contract(contracts, self.key, self.payload, self.config)
        return contracts

    def test_same_contract_allows_relocated_codec(self):
        contracts = self.register()
        payload = copy.deepcopy(self.payload)
        payload["codec"]["path"] = "restored/codec.pt"
        actual = check_task_contract(contracts, self.key, payload, self.config)
        self.assertEqual(actual["codec_sha256"], "codec-a")
        self.assertEqual(len(contracts), 1)

    def test_different_codec_checkpoint_is_rejected(self):
        contracts = self.register()
        payload = copy.deepcopy(self.payload)
        payload["codec"]["sha256"] = "codec-b"
        with self.assertRaisesRegex(ValueError, "codec_sha256"):
            check_task_contract(contracts, self.key, payload, self.config)

    def test_sampling_budgets_and_repeats_may_change(self):
        contracts = self.register()
        config = self.config | {
            "gradient_samples": 8192, "reference_samples": 65536,
            "eval_samples": 32768, "metric_samples": 4096, "repeats": 8,
        }
        actual = check_task_contract(contracts, self.key, self.payload, config)
        self.assertEqual(actual, contracts[self.key])

    def test_changed_step_list_is_rejected(self):
        contracts = self.register()
        config = self.config | {"steps": [2e-4, 3e-4, 1e-3]}
        payload = copy.deepcopy(self.payload)
        payload["methods"]["analytic"]["steps"][0]["fraction"] = 2e-4
        with self.assertRaisesRegex(ValueError, "steps"):
            check_task_contract(contracts, self.key, payload, config)

    def test_method_steps_must_match_declared_steps(self):
        payload = copy.deepcopy(self.payload)
        payload["methods"]["analytic"]["steps"][0]["fraction"] = 2e-4
        with self.assertRaisesRegex(ValueError, "step fractions within task"):
            check_task_contract({}, self.key, payload, self.config)

    def test_physical_contract_changes_are_rejected(self):
        for field, value in (("noise_std", 0.4), ("rate", 0.5), ("ebno_db", 10.0)):
            with self.subTest(field=field):
                contracts = self.register()
                payload = self.payload | {field: value}
                with self.assertRaisesRegex(ValueError, field):
                    check_task_contract(contracts, self.key, payload, self.config)

    def test_codec_configuration_change_is_rejected(self):
        contracts = self.register()
        payload = copy.deepcopy(self.payload)
        payload["codec_config"]["encoder_normalization"] = "none"
        with self.assertRaisesRegex(ValueError, "codec_config"):
            check_task_contract(contracts, self.key, payload, self.config)

    def test_legacy_optional_diagnostics_and_top_level_steps_may_be_absent(self):
        contracts = self.register()
        # Legacy split-reference diagnostics are irrelevant to task identity.
        # Fractions are still verifiable from the recorded method steps.
        config = {key: value for key, value in self.config.items() if key != "steps"}
        actual = check_task_contract(contracts, self.key, self.payload, config)
        self.assertEqual(actual["steps"], (1e-4, 3e-4, 1e-3))

    def test_missing_codec_hash_cannot_certify_identity(self):
        payload = copy.deepcopy(self.payload)
        del payload["codec"]["sha256"]
        with self.assertRaisesRegex(ValueError, "Missing codec checkpoint SHA"):
            check_task_contract({}, self.key, payload, self.config)


if __name__ == "__main__":
    unittest.main()
