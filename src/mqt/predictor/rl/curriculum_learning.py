import logging
import sys
from pathlib import Path

import pandas as pd
import json

from mqt.predictor import reward
from predictor import Predictor

logger = logging.getLogger("mqt-predictor")



def create_curriculum_with_equal_weights():
    logger.info("📊 Generating curriculum with equal weights...")

    weights = {
        #"num_qubits_norm": 1.0,
        "log_gate_count_norm": 1.0,
        #"log_avg_hopcount_norm": 1.0,
        #"max_degree_norm": 1.0,
        #"min_degree_norm": 1.0,
        #"log_adj_std_norm": 1.0,
    }

    # Normalize weights
    total_weight = sum(weights.values())
    for key in weights:
        weights[key] /= total_weight

    file_path = Path(__file__).resolve().parent / "ig_circuit_complexity_metrics_final.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find metrics file at {file_path}")

    df = pd.read_csv(file_path)
    df["complexity_score"] = sum(df[k] * w for k, w in weights.items())
    df["complexity_bin"] = pd.qcut(
        df["complexity_score"], q=5, labels=["very_easy", "easy", "medium", "hard", "very_hard"]
    )

    output_path = Path("curriculum_metrics_equal.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Curriculum saved to: {output_path.resolve()}")
    return output_path


def run_training_on_equal_curriculum(curriculum_path: Path):
    logger.info("🚀 Starting training using equal-weighted curriculum...")

    rl_pred = Predictor(
        figure_of_merit="expected_fidelity",
        device_name="ibm_washington",
        use_curriculum=True,
        curriculum_df_path=str(curriculum_path)
    )

    rl_pred.train_model(
        timesteps=20480,
        trained=0,
        verbose=2,
        save_name = "curr"
    )

    # Export timing stats
    #rl_pred.env.export_action_timings("action_timings.json")

if __name__ == "__main__":
    curriculum_path = create_curriculum_with_equal_weights()
    run_training_on_equal_curriculum(curriculum_path)
