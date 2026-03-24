"""Shared runner for capability cartography experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .adapters import AgentOverlayAdapter, GPT1WindTunnelAdapter, NotebookSubstrateAdapter
from .boundary import BoundaryAnalyzer
from .compressibility import CompressibilityStack
from .descriptors import TaskDescriptorExtractor
from .schemas import (
    ArtifactBundle,
    CapabilitySnapshot,
    CapabilityTrajectory,
    ExperimentSpec,
    InterventionConfig,
)


class CapabilityCartographyRunner:
    """Coordinates descriptor extraction, compression, and boundary analysis."""

    def __init__(
        self,
        *,
        substrate_adapter: NotebookSubstrateAdapter | None = None,
        wind_tunnel_adapter: GPT1WindTunnelAdapter | None = None,
        agent_adapter: AgentOverlayAdapter | None = None,
    ):
        self.substrate_adapter = substrate_adapter or NotebookSubstrateAdapter()
        self.wind_tunnel_adapter = wind_tunnel_adapter or GPT1WindTunnelAdapter()
        self.agent_adapter = agent_adapter or AgentOverlayAdapter()
        self.descriptor_extractor = TaskDescriptorExtractor()
        self.compressibility = CompressibilityStack()
        self.boundary = BoundaryAnalyzer()

    def run_text_experiment(
        self,
        spec: ExperimentSpec,
        intervention: InterventionConfig,
        *,
        text: str,
        retrieval_context: str = "",
        metric_series: Optional[Sequence[Dict[str, float]]] = None,
        export_dir: str | Path | None = None,
    ) -> ArtifactBundle:
        snapshots: List[CapabilitySnapshot] = []
        metric_series = metric_series or self._default_metric_series(text, intervention)
        for step, metrics in enumerate(metric_series, start=1):
            descriptor = self.descriptor_extractor.extract_text_descriptor(
                text,
                task_name=spec.task_name,
                benchmark_label=spec.benchmark_label,
                substrate=spec.substrate,
                realism_level=spec.realism_level,
                metadata=spec.metadata,
                retrieval_context=retrieval_context,
            )
            profile = self.compressibility.profile_text(
                text,
                predictive_loss=metrics.get("loss_proxy"),
            )
            snapshots.append(
                CapabilitySnapshot(
                    step=step,
                    metrics=metrics,
                    descriptor=descriptor,
                    compressibility=profile,
                    notes={"intervention": intervention.to_dict()},
                )
            )

        trajectory = CapabilityTrajectory(
            experiment_id=spec.experiment_id,
            substrate=spec.substrate,
            intervention_config=intervention.to_dict(),
            snapshots=snapshots,
        )
        trajectory.boundary_events = self.boundary.detect_events(snapshots, metric="capability_score")
        trajectory.fitted_boundaries = [self.boundary.fit_threshold(snapshots, metric="capability_score")]

        bundle = ArtifactBundle(spec=spec, trajectory=trajectory)
        bundle.narrative = self.agent_adapter.narrate(bundle.to_dict())
        if export_dir is not None:
            bundle.export_path = self.export(bundle, export_dir=export_dir)
        return bundle

    def profile_gpt1_wind_tunnel(
        self,
        *,
        prompt: str,
        intervention: InterventionConfig,
        export_dir: str | Path | None = None,
    ) -> ArtifactBundle:
        architecture = intervention.architecture
        metrics = self.wind_tunnel_adapter.dry_run_metrics(
            prompt=prompt,
            vocab_size=int(architecture.get("vocab_size", 96)),
            d_model=int(architecture.get("d_model", 64)),
            num_heads=int(architecture.get("num_heads", 4)),
            num_layers=int(architecture.get("num_layers", 2)),
            d_ff=int(architecture.get("d_ff", 128)),
            max_seq_len=int(intervention.context_geometry.get("max_seq_len", 64)),
        )
        metric_series = [
            {
                "capability_score": float(np.tanh(metrics["capacity_proxy"] / 5000.0)),
                "loss_proxy": float(max(0.1, 1.2 - np.tanh(metrics["logit_std"]))),
                "retrieval_dependence": float(intervention.retrieval.get("enabled", False)),
            },
            {
                "capability_score": float(min(1.0, np.tanh(metrics["capacity_proxy"] / 3500.0) + 0.08)),
                "loss_proxy": float(max(0.05, 0.95 - np.tanh(metrics["logit_std"]))),
                "retrieval_dependence": float(intervention.retrieval.get("enabled", False)),
            },
        ]
        spec = ExperimentSpec(
            experiment_id="gpt1-wind-tunnel",
            substrate="gpt1-from-sutskever30",
            task_name="wind_tunnel_probe",
            benchmark_label="gpt1_dry_run",
            realism_level="semi_synthetic",
            objective_type=str(intervention.objective.get("loss_type", "next_token")),
            model_family="gpt1",
            intervention_axes=list(intervention.flattened().keys()),
            metadata={"prompt_length": len(prompt)},
        )
        return self.run_text_experiment(
            spec,
            intervention,
            text=prompt,
            retrieval_context="",
            metric_series=metric_series,
            export_dir=export_dir,
        )

    @staticmethod
    def export(bundle: ArtifactBundle, *, export_dir: str | Path) -> str:
        path = Path(export_dir)
        path.mkdir(parents=True, exist_ok=True)
        export_path = path / f"{bundle.spec.experiment_id}.json"
        export_path.write_text(json.dumps(bundle.to_dict(), indent=2))
        return str(export_path)

    @staticmethod
    def _default_metric_series(text: str, intervention: InterventionConfig) -> List[Dict[str, float]]:
        length_factor = min(len(text) / 200.0, 1.0)
        retrieval_penalty = float(intervention.retrieval.get("distractor_density", 0.0))
        context_bonus = min(float(intervention.context_geometry.get("answer_position", 0)) / 100.0, 0.2)
        base = max(0.1, 0.45 + length_factor + context_bonus - retrieval_penalty)
        return [
            {
                "capability_score": float(min(base, 1.0)),
                "loss_proxy": float(max(0.1, 1.1 - base)),
                "retrieval_dependence": float(intervention.retrieval.get("enabled", False)),
            },
            {
                "capability_score": float(min(base + 0.18, 1.0)),
                "loss_proxy": float(max(0.05, 0.95 - base)),
                "retrieval_dependence": float(intervention.retrieval.get("enabled", False)),
            },
            {
                "capability_score": float(min(base + 0.3, 1.0)),
                "loss_proxy": float(max(0.02, 0.75 - base)),
                "retrieval_dependence": float(intervention.retrieval.get("enabled", False)),
            },
        ]
