"""Basic verification for the Capability Cartography Layer."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from capability_cartography.boundary import BoundaryAnalyzer
from capability_cartography.compressibility import CompressibilityStack
from capability_cartography.descriptors import TaskDescriptorExtractor
from capability_cartography.runner import CapabilityCartographyRunner
from capability_cartography.schemas import CapabilitySnapshot, ExperimentSpec, InterventionConfig


class CapabilityCartographyTests(unittest.TestCase):
    def test_descriptor_extraction_text(self):
        extractor = TaskDescriptorExtractor()
        descriptor = extractor.extract_text_descriptor(
            "If Alice retrieves the right passage, then the answer follows.",
            task_name="qa",
            benchmark_label="unit",
            substrate="test",
            retrieval_context="retrieves answer passage",
        )
        self.assertGreaterEqual(descriptor.retrieval_geometry["retrieval_dependency_score"], 0.0)
        self.assertEqual(descriptor.cognitive_operations["logical_deduction"], 1.0)

    def test_compressibility_stack_array(self):
        stack = CompressibilityStack()
        profile = stack.profile_array([[1, 0, 1], [1, 0, 1]])
        self.assertIn("gzip_ratio", profile.surface)
        self.assertGreaterEqual(profile.structural["effective_params"], 1)

    def test_boundary_detection(self):
        analyzer = BoundaryAnalyzer()
        snapshots = [
            CapabilitySnapshot(step=1, metrics={"capability_score": 0.2}, descriptor=None, compressibility=None),  # type: ignore[arg-type]
            CapabilitySnapshot(step=2, metrics={"capability_score": 0.5}, descriptor=None, compressibility=None),  # type: ignore[arg-type]
            CapabilitySnapshot(step=3, metrics={"capability_score": 0.85}, descriptor=None, compressibility=None),  # type: ignore[arg-type]
        ]
        events = analyzer.detect_events(snapshots, metric="capability_score")
        self.assertGreaterEqual(len(events), 1)

    def test_runner_exports(self):
        runner = CapabilityCartographyRunner()
        intervention = InterventionConfig(
            architecture={"d_model": 64, "num_heads": 4, "num_layers": 2, "d_ff": 128, "vocab_size": 96},
            objective={"loss_type": "next_token"},
            retrieval={"enabled": True, "distractor_density": 0.2},
            context_geometry={"answer_position": 32, "max_seq_len": 64},
        )
        spec = ExperimentSpec(
            experiment_id="unit-demo",
            substrate="unit-test",
            task_name="qa",
            benchmark_label="unit",
            realism_level="synthetic",
            objective_type="next_token",
            model_family="unit",
        )
        with TemporaryDirectory() as temp_dir:
            bundle = runner.run_text_experiment(
                spec,
                intervention,
                text="The model must retrieve a fact and reason over it.",
                retrieval_context="fact retrieval",
                export_dir=Path(temp_dir),
            )
            self.assertIsNotNone(bundle.export_path)
            self.assertTrue(Path(bundle.export_path).exists())


if __name__ == "__main__":
    unittest.main()
