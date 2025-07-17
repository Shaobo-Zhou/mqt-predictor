"""This module contains the Predictor class, which is used to predict the most suitable compilation pass sequence for a given quantum circuit."""

from __future__ import annotations

import logging
import os
import sys
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING
from pathlib import Path
from collections import deque
from copy import deepcopy
import csv

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure, Logger, KVWriter, HumanOutputFormat, TensorBoardOutputFormat


from mqt.predictor import reward, rl

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

logger = logging.getLogger("mqt-predictor")
PATH_LENGTH = 260


""" class OffsetCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, name_prefix, offset=0, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.offset = offset

    def _on_step(self) -> bool:
        total_steps = self.num_timesteps + self.offset
        if total_steps % self.save_freq == 0:
            path = f"{self.save_path}/{self.name_prefix}_{total_steps}_steps.zip"
            #self.model.save(path)
            if self.verbose > 0:
                print(f"✅ Saved checkpoint: {path}")
        return True
 """
class CurriculumProgressionCallback(BaseCallback):
    def __init__(
        self,
        env: rl.PredictorEnv,
        threshold: float = 0.3,
        check_freq: int = 50,
        verbose: int = 1,
        save_dir: str = "./checkpoints/curriculum_progression",
        thresholds_by_level: dict[int, float] = None,
        margin: float = 0.02,
        near_threshold_limit: int = 2,
        final_level_no_improve_limit: int = 3,
    ):
        super().__init__(verbose)
        self.env = env
        self.default_threshold = threshold
        self.check_freq = check_freq
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.margin = margin
        self.near_threshold_limit = near_threshold_limit
        self.final_level_no_improve_limit = final_level_no_improve_limit

        

        self.max_level = self.env.max_difficulty_level
        self.thresholds_by_level = thresholds_by_level or {
            0: 0.723,
            1: 0.354,
            2: 0.197,
            3: 0.101,
            4: 0.017,
        }

        self.near_threshold_count = 0
        self.episode_rewards = []
        self.stop_training_now = False
        self.best_final_level_reward = float("-inf")
        self.final_level_no_improve_count = 0

    def _on_step(self) -> bool:
        if self.stop_training_now:
            return False
        
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    rewards = info["episode"]["r"]
                    self.episode_rewards.append(rewards)
        return True


    def _on_rollout_end(self) -> None:
        if len(self.episode_rewards) < self.check_freq:
            if self.verbose:
                logger.info(f"[Curriculum] ⏳ Waiting for more episodes... ({len(self.episode_rewards)}/{self.check_freq})")
            return

        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
        current_level = self.env.current_difficulty_level
        threshold = self.thresholds_by_level.get(current_level, self.default_threshold)

        if self.verbose:
            logger.info(f"[Curriculum] 📊 Avg reward: {avg_reward:.4f} | Threshold: {threshold:.4f}")

        # --- Final level saturation logic ---
        if current_level == self.max_level:
            if avg_reward > self.best_final_level_reward + 1e-5:
                self.best_final_level_reward = avg_reward
                self.final_level_no_improve_count = 0
                logger.info(f"[Curriculum] 📈 New best reward at final level: {avg_reward:.4f}")
            else:
                self.final_level_no_improve_count += 1
                logger.info(f"[Curriculum] 💤 Final level not improving ({self.final_level_no_improve_count}/{self.final_level_no_improve_limit})")
                if self.final_level_no_improve_count >= self.final_level_no_improve_limit:
                    logger.info("✅ Final level saturated — stopping training.")
                    self.stop_training_now = True
        else:
            # --- Promotion logic ---
            promote = False
            if avg_reward >= threshold:
                promote = True
            elif threshold - self.margin <= avg_reward < threshold:
                self.near_threshold_count += 1
                if self.verbose:
                    logger.info(f"[Curriculum] ⚠️ Near threshold ({self.near_threshold_count}/{self.near_threshold_limit})")
                promote = self.near_threshold_count >= self.near_threshold_limit

            if promote:
                updated = self.env.increase_curriculum_difficulty()
                if updated:
                    logger.info(f"[Curriculum] 🔼 Promoted to level {self.env.current_difficulty_level}")
                    save_path = os.path.join(self.save_dir, f"model_level_{self.env.current_difficulty_level}.zip")
                    self.model.save(save_path)
                    logger.info(f"[Curriculum] 💾 Model saved: {save_path}")
                    self.near_threshold_count = 0

        self.episode_rewards = []


class SaturationCurriculumCallback(BaseCallback):
    def __init__(
        self,
        env: rl.PredictorEnv,
        check_freq: int = 50,
        verbose: int = 1,
        save_dir: str = "./checkpoints/curriculum_saturation",
        saturation_patience: int = 3,
        thresholds_by_level: dict[int, float] = None,
        save_best_checkpoint: bool = False,  
    ):
        super().__init__(verbose)
        self.env = env
        self.check_freq = check_freq
        self.save_dir = save_dir
        self.saturation_patience = saturation_patience
        self.max_level = self.env.max_difficulty_level
        self.save_best_checkpoint = save_best_checkpoint  

        self.thresholds_by_level = thresholds_by_level or {
            0: 0.723,
            1: 0.354,
            2: 0.197,
            3: 0.101,
            4: 0.017,
        }

        os.makedirs(self.save_dir, exist_ok=True)

        # Tracking improvement for each level
        self.episode_rewards = []
        self.best_reward = float("-inf")
        self.no_improve_count = 0
        self.stop_training_now = False

    def _on_step(self) -> bool:
        if self.stop_training_now:
            return False
        
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    rewards = info["episode"]["r"]
                    self.episode_rewards.append(rewards)
        return True

    def _on_rollout_end(self) -> None:
        if len(self.episode_rewards) < self.check_freq:
            if self.verbose:
                logger.info(f"[Curriculum] ⏳ Waiting for more episodes... ({len(self.episode_rewards)}/{self.check_freq})")
            return

        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
        if self.verbose:
            logger.info(f"[Curriculum] 📊 Avg reward: {avg_reward:.4f}")

        level = self.env.current_difficulty_level
        improved = avg_reward > self.best_reward + 1e-5

        if improved:
            self.best_reward = avg_reward
            self.no_improve_count = 0
            logger.info(f"[SaturationCurriculum] 📈 New best reward at level {level}: {avg_reward:.4f}")

            if self.save_best_checkpoint:
                save_path = os.path.join(self.save_dir, f"model_best_level_{level}.zip")
                self.model.save(save_path)
                logger.info(f"[SaturationCurriculum] 💾 Saved best model at level {level}: {save_path}")
        else:
            self.no_improve_count += 1
            logger.info(f"[SaturationCurriculum] 💤 No improvement ({self.no_improve_count}/{self.saturation_patience})")

        if self.no_improve_count >= self.saturation_patience:
            if level < self.max_level:
                updated = self.env.increase_curriculum_difficulty()
                if updated:
                    logger.info(f"[SaturationCurriculum] 🔼 Promoted to level {self.env.current_difficulty_level}")
                    if not self.save_best_checkpoint:
                        save_path = os.path.join(self.save_dir, f"model_level_{self.env.current_difficulty_level}.zip")
                        self.model.save(save_path)
                        logger.info(f"[SaturationCurriculum] 💾 Model saved on promotion: {save_path}")
                    self.best_reward = float("-inf")
                    self.no_improve_count = 0
            else:
                logger.info("✅ Final level saturated — stopping training.")
                self.stop_training_now = True

        self.episode_rewards = []

class ActionEffectivenessEarlyStoppingCallback(BaseCallback):
    def __init__(
        self,
        env,
        save_dir: str = "./action_stats",
        verbose: int = 0,
        reward_patience: int = 5,
        reward_delta: float = 1e-4,
        reward_threshold: float = None,
    ):
        super().__init__(verbose)
        self.env = env
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.rollout_count = 0
        self.verbose = verbose

        # Early stopping config
        self.reward_patience = reward_patience
        self.reward_delta = reward_delta
        self.reward_threshold = reward_threshold
        self.best_mean_reward = -np.inf
        self.best_mean_final_step = -np.inf
        self.no_improve_count = 0
        self.stop_training_now = False
        self.episode_rewards = []  
        self.final_step_rewards = []

    def _on_step(self) -> bool:
        # Collect episode rewards as they finish
        if self.stop_training_now:
            return False
        infos = self.locals.get("infos")
        rewards = self.locals.get("rewards")
        if infos and rewards is not None:
            for env_idx, info in enumerate(infos):
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.final_step_rewards.append(rewards[env_idx])
        return True

    def _on_rollout_end(self) -> None:
        self.rollout_count += 1

        # Save action effectiveness log
        filename = f"action_effectiveness_rollout_{self.rollout_count:05d}.json"
        save_path = self.save_dir / filename
        if self.verbose:
            logger.info(f"[ActionStats] 💾 Saving to {save_path}")
        self.env.export_action_effectiveness(str(save_path))

        checkpoint_path = self.save_dir / "latest_checkpoint.zip"
        if hasattr(self, 'model'):
            self.model.save(str(checkpoint_path))
            if self.verbose:
                logger.info(f"[ActionStats] 💾 Checkpoint saved to {checkpoint_path}")
        else:
            logger.warning("[ActionStats] No model attached to callback for checkpointing.")

        # Early stopping logic (fidelity/final step only)
        if not self.final_step_rewards:
            if self.verbose:
                logger.info(f"[ActionStats] ⚠️ No finished episodes in this rollout.")
            return

        mean_final_step = sum(self.final_step_rewards) / len(self.final_step_rewards)

        if self.verbose:
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards) if self.episode_rewards else None
            logger.info(f"[ActionStats] 📊 Mean episode reward (this rollout): {avg_reward:.4f}" if avg_reward is not None else "[ActionStats] 📊 No episode rewards.")
            logger.info(f"[ActionStats] 🏁 Mean final step reward (fidelity, this rollout): {mean_final_step:.6f}")

        # --- Early stopping on final step reward (fidelity) only ---
        if mean_final_step > self.best_mean_final_step + self.reward_delta:
            self.best_mean_final_step = mean_final_step
            self.no_improve_count_final_step = 0
            if self.verbose:
                logger.info(f"[ActionStats] 🆕 Improved mean final step reward: {mean_final_step:.6f}")
        else:
            self.no_improve_count_final_step += 1
            if self.verbose:
                logger.info(f"[ActionStats] 😴 No improvement on final step ({self.no_improve_count_final_step}/{self.reward_patience})")
        if (self.reward_threshold is not None) and (mean_final_step >= self.reward_threshold):
            logger.info(f"[ActionStats] 🎯 Early stopping: final step reward threshold {self.reward_threshold} reached!")
            self.stop_training_now = True
        if self.no_improve_count_final_step >= self.reward_patience:
            logger.info(f"[ActionStats] ⏹️ Early stopping: no improvement for {self.reward_patience} rollouts (final step)!")
            self.stop_training_now = True

        # Reset episode rewards for the next rollout
        self.episode_rewards = []
        self.final_step_rewards = []

    def _on_training_end(self):
        csv_path = self.save_dir / "final_step_rewards.csv"
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["final_step_reward"])
            for r in self.final_step_rewards:
                writer.writerow([r])
        if self.verbose:
            logger.info(f"[ActionStats] Training ended. Best mean reward: {self.best_mean_final_step:.4f}")

                  

class Predictor:
    """The Predictor class is used to train a reinforcement learning model for a given figure of merit and device such that it acts as a compiler."""

    def __init__(
        self, figure_of_merit: reward.figure_of_merit, device_name: str, logger_level: int = logging.INFO, use_curriculum: bool = False,
        curriculum_df_path: str = "curriculum_metrics_optuna.csv"
    ) -> None:
        """Initializes the Predictor object."""
        logger.setLevel(logger_level)

        self.env = rl.PredictorEnv(reward_function=figure_of_merit, device_name=device_name)
        if use_curriculum:
            df_curriculum = pd.read_csv(curriculum_df_path)
            self.env.set_curriculum_data(df_curriculum, enable_sampling=True)
        self.device_name = device_name
        self.figure_of_merit = figure_of_merit

    # def compile_as_predicted(
    #     self,
    #     qc: QuantumCircuit,
    #     file_name: str,
    # ) -> tuple[QuantumCircuit, list[str]]:
        """ Compiles a given quantum circuit such that the given figure of merit is maximized by using the respectively trained optimized compiler.

        Arguments:
            qc: The quantum circuit to be compiled or the path to a qasm file containing the quantum circuit.

        Returns:
            A tuple containing the compiled quantum circuit and the compilation information. If compilation fails, False is returned. """
       
        # folder = rl.helper.get_path_trained_model()
        # model_path = folder / file_name

        # #trained_rl_model = rl.helper.load_model("model_" + self.figure_of_merit + "_" + self.device_name)
        # trained_rl_model = rl.helper.load_model(str(model_path))

        # obs, _ = self.env.reset(qc, seed=0)

        # used_compilation_passes = []
        # terminated = False
        # truncated = False
        # while not (terminated or truncated):
        #     action_masks = get_action_masks(self.env)
        #     action, _ = trained_rl_model.predict(obs, action_masks=action_masks)
        #     action = int(action)
        #     action_item = self.env.action_set[action]
        #     used_compilation_passes.append(action_item["name"])
        #     obs, _reward_val, terminated, truncated, _info = self.env.step(action)

        # if not self.env.error_occurred:
        #     return self.env.state, _reward_val, used_compilation_passes

        # msg = "Error occurred during compilation."
        # raise RuntimeError(msg) 
    
    def compile_as_predicted(
        self,
        qc: QuantumCircuit,
        file_name: str,
    ) -> tuple[QuantumCircuit, float, list[str]]:
        """Compiles a given quantum circuit using a trained RL model.

        Args:
            qc: Quantum circuit or path to QASM file.
            file_name: Name of the trained model file to use.

        Returns:
            A tuple of the compiled quantum circuit, reward, and the list of used compilation passes.
        """
        folder = rl.helper.get_path_trained_model()
        model_path = folder / file_name
        trained_rl_model = rl.helper.load_model(str(model_path))

        obs, _ = self.env.reset(qc, seed=0)

        used_compilation_passes = []
        terminated = False
        truncated = False

        step_count = 0
        max_steps = 200
        recent_actions = deque(maxlen=8)
        blocked_actions = set()

        unstable_actions = {"AIRouting", "AICliffordSynthesis", "AILinearFunctionSynthesis", "AIPermutationSynthesis"}
        for idx, action in self.env.action_set.items():
            if action["name"] in unstable_actions:
                blocked_actions.add(idx)

        while not (terminated or truncated) and step_count < max_steps:
            action_masks = get_action_masks(self.env)
            # Mask out repeated actions
            if blocked_actions:
                action_masks = deepcopy(action_masks)
                for idx in blocked_actions:
                    action_masks[idx] = False

            action, _ = trained_rl_model.predict(obs, action_masks=action_masks, deterministic=True)
            action = int(action)

            recent_actions.append(action)

            max_cycle_length = 4

            def is_cycle(lst, k):
                if len(lst) < 2*k:
                    return False
                return lst[-2*k:-k] == lst[-k:]
            
            for k in range(2, max_cycle_length+1):
                if is_cycle(list(recent_actions), k):
                    print(f"Avoiding {k}-cycle infinite loop pattern")
                    for cyc_action in set(list(recent_actions)[-k:]):
                        blocked_actions.add(cyc_action)
                    break

            # If the same action is used consecutively, block it to avoid an infinite loop
            """ if len(recent_actions) >= 3 and all(a == action for a in list(recent_actions)[-3:]):
                print("Avoiding infinite loop")
                blocked_actions.add(action) """

            action_item = self.env.action_set[action]
            used_compilation_passes.append(action_item["name"])
            obs, _reward_val, terminated, truncated, _info = self.env.step(action)
            step_count += 1

        if not self.env.error_occurred:
            return self.env.state, _reward_val, used_compilation_passes

        msg = "Error occurred during compilation."
        raise RuntimeError(msg)

    def train_model(
        self,
        timesteps: int = 100000,
        model_name: str = "model",
        verbose: int = 2,
        test: bool = False,
        trained: int = 0,
        save_name: str = "default",
        resume_from_level: int = None,
        resume_model_path: str = None,
    ) -> None:
        """Train or resume model training with offset checkpointing."""
        n_steps = 10 if test else 2048
        progress_bar = not test

        name_prefix = f"{model_name}_{self.figure_of_merit}_{self.device_name}"
        log_dir = f"./{name_prefix}"
        checkpoint_dir = f"./checkpoints/{save_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        ckpt_path = os.path.join(checkpoint_dir, f"{name_prefix}_{trained}_steps.zip")
        logger.debug(f"🔁 Checking for checkpoint: {ckpt_path}")

        # === Set resume curriculum level ===
        if resume_from_level is not None:
            self.env.current_difficulty_level = resume_from_level
            logger.info(f"🔄 Resuming from curriculum level {resume_from_level}")

        # === Load model ===
        if resume_model_path and Path(resume_model_path).exists():
            logger.info(f"📦 Loading model from {resume_model_path}")
            model = MaskablePPO.load(
                resume_model_path,
                env=self.env,
                tensorboard_log=log_dir,
                verbose=verbose,
                device="cuda",
            )
        else:
            logger.info("🆕 No model specified for resume, starting fresh training")
            model = MaskablePPO(
                MaskableMultiInputActorCriticPolicy,
                self.env,
                verbose=verbose,
                tensorboard_log=log_dir,
                gamma=0.98,
                n_steps=n_steps,
            )

        remaining = timesteps - trained
        os.makedirs(f"./checkpoints/{save_name}", exist_ok=True)
        callback=None
        callback = ActionEffectivenessEarlyStoppingCallback(
            env=self.env,
            save_dir=f"./checkpoints/{save_name}/action_logs",          
            reward_patience=10,       
            reward_delta=1e-4,       
            verbose=1
        )
        #callback = CurriculumProgressionCallback(env=self.env, threshold=0.3, check_freq=50, save_dir=rl.helper.get_path_trained_model() / save_name)
        tb_log_name = "ppo"
        log_path = os.path.join(log_dir, save_name,"ppo") 
        new_logger = configure(folder=log_path, format_strings=["stdout", "tensorboard"])
        model.set_logger(new_logger)

        model.learn(
            total_timesteps=remaining,
            tb_log_name=tb_log_name,
            callback=callback,
            progress_bar=progress_bar,
        )

        save_path = rl.helper.get_path_trained_model() / save_name
        save_path.mkdir(parents=True, exist_ok=True)
        model.save(save_path / f"{name_prefix}")

        logger.info("✅ Final model saved.")
