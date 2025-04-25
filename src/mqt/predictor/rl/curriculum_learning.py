import logging
import sys
from pathlib import Path

import pandas as pd
import json

from mqt.predictor import reward, rl
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger("mqt-predictor")

class CurriculumProgressionCallback(BaseCallback):
    def __init__(self, env: rl.PredictorEnv, threshold=0.3, check_freq=500, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.threshold = threshold
        self.check_freq = check_freq
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])

        if len(self.episode_rewards) >= self.check_freq:
            avg_reward = sum(self.episode_rewards[-self.check_freq:]) / self.check_freq
            if self.verbose:
                logger.info(f"📊 Avg reward over last {self.check_freq} episodes: {avg_reward:.4f}")
            if avg_reward >= self.threshold:
                updated = self.env.increase_curriculum_difficulty()
                if updated:
                    self.episode_rewards = []  # reset tracking for new level
        return True
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

    file_path = Path(__file__).resolve().parent / "ig_circuit_complexity_metrics_30.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Could not find metrics file at {file_path}")

    df = pd.read_csv(file_path)
    df["complexity_score"] = sum(df[k] * w for k, w in weights.items())
    df["complexity_bin"] = pd.qcut(
        df["complexity_score"], q=5, labels=["very_easy", "easy", "medium", "hard", "very_hard"]
    )

    output_path = Path("curriculum_metrics_qubit.csv")
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

    curriculum_callback = CurriculumProgressionCallback(env=rl_pred.env, threshold=0.3, check_freq=2048)

    rl_pred.train_model(
        timesteps=100000,
        trained=0,
        verbose=2,
        save_name = "curr",
        custom_callbacks=[curriculum_callback]
    )

    # Export timing stats
    #rl_pred.env.export_action_timings("action_timings.json")

if __name__ == "__main__":
    curriculum_path = create_curriculum_with_equal_weights()
    run_training_on_equal_curriculum(curriculum_path)
