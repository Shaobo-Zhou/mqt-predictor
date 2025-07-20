import logging
import sys
from pathlib import Path

import pandas as pd
import json

from mqt.predictor import reward, rl
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger("mqt-predictor")

def create_curriculum_with_equal_weights():
    logger.info("📊 Generating curriculum with equal weights...")

    weights = {
        "num_qubits_norm": 1.0,
        "depth_norm": 1.0,
        #"log_gate_count_norm": 1.0,
        #"log_avg_hopcount_norm": 1.0,
        #"max_degree_norm": 1.0,
        #"min_degree_norm": 1.0,
        #"log_adj_std_norm": 1.0,
        "gate_count_norm": 1.0,
    }

    # Normalize weights
    total_weight = sum(weights.values())
    for key in weights:
        weights[key] /= total_weight

    file_path = Path(__file__).resolve().parent / "metrics_new_indep.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find metrics file at {file_path}")

    df = pd.read_csv(file_path)

    for col in ["num_qubits", "depth", "gate_count"]:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val - min_val > 0:
            df[f"{col}_norm"] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[f"{col}_norm"] = 0.0
            
    df["complexity_score"] = sum(df[k] * w for k, w in weights.items())
    df["complexity_bin"] = pd.qcut(
        df["complexity_score"], q=5, labels=["very_easy", "easy", "medium", "hard", "very_hard"]
    )

    output_path = Path("curriculum_metrics_combined.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Curriculum saved to: {output_path.resolve()}")
    return output_path


def run_training_on_equal_curriculum(curriculum_path: Path):
    logger.info("🚀 Starting training using equal-weighted curriculum...")

    rl_pred = rl.Predictor(
        figure_of_merit="expected_fidelity",
        device_name="ibm_washington",
        use_curriculum=True,
        curriculum_df_path=str(curriculum_path)
    )


    rl_pred.train_model(
        timesteps=100000,
        trained=0,
        verbose=2,
        save_name = "curr_new_new",
        curriculum=True,
        #resume_from_level=1,
        #resume_model_path="./checkpoints/curr_new_new/model_best_level_1.zip"
    )

if __name__ == "__main__":
    curriculum_path = create_curriculum_with_equal_weights()
    curriculum_path = Path("curriculum_metrics_combined.csv")
    run_training_on_equal_curriculum(curriculum_path)
