"""Runnable demo for the Capability Cartography Layer."""

from __future__ import annotations

import json
from pathlib import Path

from .runner import CapabilityCartographyRunner
from .schemas import ExperimentSpec, InterventionConfig


def main() -> None:
    runner = CapabilityCartographyRunner()

    intervention = InterventionConfig(
        architecture={"d_model": 64, "num_heads": 4, "num_layers": 2, "d_ff": 128, "vocab_size": 96},
        objective={"loss_type": "next_token"},
        data_regime={"dataset_type": "semi_synthetic", "compressibility_target": 0.72},
        retrieval={"enabled": True, "distractor_density": 0.35, "position": "middle"},
        context_geometry={"answer_position": 48, "max_seq_len": 64},
        interpretability={"activation_patching": False},
    )

    spec = ExperimentSpec(
        experiment_id="capability-cartography-demo",
        substrate="sutskever-30-implementations",
        task_name="descriptor_failure_probe",
        benchmark_label="demo_reasoning_probe",
        realism_level="semi_synthetic",
        objective_type="next_token",
        model_family="gpt1-compatible",
        intervention_axes=list(intervention.flattened().keys()),
        metadata={"source": "Capability Cartography Layer demo"},
    )

    text = (
        "A proof sketch is placed in the middle of a long context with several distractor passages. "
        "The model must retrieve the relevant lemma, reason through two linked steps, and identify the answer."
    )
    retrieval_context = (
        "Distractor passage alpha. Distractor passage beta. Relevant lemma about linked reasoning steps. "
        "Additional irrelevant passage about unrelated retrieval documents."
    )

    repo_root = Path(__file__).resolve().parents[1]
    artifacts_dir = repo_root / "artifacts"

    bundle = runner.run_text_experiment(
        spec,
        intervention,
        text=text,
        retrieval_context=retrieval_context,
        export_dir=artifacts_dir,
    )

    wind_tunnel_bundle = runner.profile_gpt1_wind_tunnel(
        prompt="the capability atlas predicts a threshold",
        intervention=intervention,
        export_dir=artifacts_dir,
    )

    print("Capability Cartography Demo")
    print(json.dumps(bundle.to_dict(), indent=2)[:1200])
    print()
    print("Wind Tunnel Summary")
    print(json.dumps(wind_tunnel_bundle.to_dict(), indent=2)[:1200])


if __name__ == "__main__":
    main()
