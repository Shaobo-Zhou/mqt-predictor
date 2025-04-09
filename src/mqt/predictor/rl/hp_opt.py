import os
import sys
import logging
from typing import TYPE_CHECKING

import optuna
import pandas as pd
import numpy as np
from pathlib import Path
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.logger import configure, Logger, HumanOutputFormat, TensorBoardOutputFormat
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from mqt.predictor import reward, rl
from predictor import Predictor, OffsetCheckpointCallback  # Your existing classes

import logging

logger = logging.getLogger("mqt-predictor")

class OptunaPruningCallback(BaseCallback):
    def __init__(self, trial, check_freq: int = 500, log: logging.Logger = None, verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self.check_freq = check_freq
        self.rewards = []
        self._log = log or logging.getLogger(__name__)  # ← use a different name

    def _on_step(self) -> bool:
        self._log.debug(f"[Callback] Step {self.n_calls} reached")
        if "rollout/ep_rew_mean" in self.model.logger.name_to_value:
            reward_val = self.model.logger.name_to_value["rollout/ep_rew_mean"]
            if reward_val is not None:
                self.rewards.append(reward_val)

        if self.n_calls % self.check_freq == 0 and self.rewards:
            avg_reward = float(np.mean(self.rewards[-10:]))
            self._log.info(f"📈 [Step {self.n_calls}] Reporting avg_reward = {avg_reward:.4f} to Optuna")
            self.trial.report(avg_reward, self.n_calls)

            if self.trial.should_prune():
                self._log.warning(f"⛔ [Step {self.n_calls}] Trial pruned with avg_reward = {avg_reward:.4f}")
                raise optuna.exceptions.TrialPruned()

        return True




def objective(trial):
    # Sample weights for normalized features
    weights = {
        "num_qubits_norm": trial.suggest_float("w_qubits", 0, 1),
        "log_gate_count_norm": trial.suggest_float("w_gates", 0, 1),
        "log_avg_hopcount_norm": trial.suggest_float("w_hop", 0, 1),
        "max_degree_norm": trial.suggest_float("w_maxdeg", 0, 1),
        "min_degree_norm": trial.suggest_float("w_mindeg", 0, 1),
        "log_adj_std_norm": trial.suggest_float("w_adjstd", 0, 1),
    }

    # Normalize weights
    total_weight = sum(weights.values())
    for key in weights:
        weights[key] /= total_weight

    # Load and compute complexity score
    file_path = Path(__file__).resolve().parent / "ig_circuit_complexity_metrics_final.csv"
    df = pd.read_csv(file_path)
    df["complexity_score"] = sum(df[k] * w for k, w in weights.items())
    # Bin into curriculum stages using quantiles
    df["complexity_bin"] = pd.qcut(
        df["complexity_score"], q=5, labels=["very_easy", "easy", "medium", "hard", "very_hard"]
    )
    df.to_csv("curriculum_metrics_optuna.csv", index=False)

    # Setup RL training with pruning callback
    rl_pred = Predictor("expected_fidelity", "ibm_washington", use_curriculum=True)
    pruning_cb = OptunaPruningCallback(trial, check_freq=500, log=logging.getLogger("mqt-predictor"))

    try:
        rl_pred.train_model(
            timesteps=5000,
            trained=0,
            custom_callbacks=[pruning_cb],
        )
    except optuna.exceptions.TrialPruned:
        raise

    final_reward = pruning_cb.rewards[-1] if pruning_cb.rewards else 0.0
    return final_reward

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, timeout=20000)

    print("✅ Best Trial:")
    print(study.best_trial)
