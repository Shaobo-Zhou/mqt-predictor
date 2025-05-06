"""This module contains the Predictor class, which is used to predict the most suitable compilation pass sequence for a given quantum circuit."""

from __future__ import annotations

import logging
import os
import sys
import pandas as pd
from typing import TYPE_CHECKING

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
    def __init__(self, env: rl.PredictorEnv, threshold=0.3, check_freq=50, verbose=1):
        super().__init__(verbose)
        self.env = env
        self.default_threshold = threshold
        self.thresholds_by_level = {
            0: 0.723,   # very_easy
            1: 0.354,   # easy
            2: 0.197,   # medium
            3: 0.101,   # hard
            4: 0.017    # very_hard
        }
        self.check_freq = check_freq
        self.margin = 0.02
        self.near_threshold_limit = 2
        self.near_threshold_count = 0           
        self.episode_rewards = []

    def _on_step(self) -> bool:
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if "episode" in info:
                    rewards = info["episode"]["r"]
                    #logger.info(f"Current episode reward is {rewards}")
                    self.episode_rewards.append(rewards)
        return True
    def _on_rollout_end(self) -> None:
        if len(self.episode_rewards) >= self.check_freq:
            avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)

            current_level = self.env.current_difficulty_level
            threshold = self.thresholds_by_level.get(current_level, self.default_threshold)

            if self.verbose:
                logger.info(f"[Curriculum] 📊 Avg reward: {avg_reward:.4f} | Threshold: {threshold:.4f}")

            if avg_reward >= threshold:
                promote = True
            elif threshold - self.margin <= avg_reward < threshold:
                self.near_threshold_count += 1
                if self.verbose:
                    logger.info(f"[Curriculum] ⚠️ Near threshold ({self.near_threshold_count}/{self.near_threshold_limit})")
                promote = self.near_threshold_count >= self.near_threshold_limit
            else:
                self.near_threshold_count = 0
                promote = False

            if promote:
                updated = self.env.increase_curriculum_difficulty()
                if updated and self.verbose:
                    logger.info(f"[Curriculum] 🔼 Promoted to difficulty level {self.env.current_difficulty_level}")
                self.near_threshold_count = 0
        else:
            if self.verbose:
                logger.info(f"[Curriculum] ⏳ Waiting for more episodes... ({len(self.episode_rewards)}/{self.check_freq})")

        self.episode_rewards = []



class OffsetLogger(Logger):
    def __init__(self, trained_offset, folder, format_strings):
        os.makedirs(folder, exist_ok=True)
        output_formats = []

        for fmt in format_strings:
            if fmt == "stdout":
                output_formats.append(HumanOutputFormat(sys.stdout))
            elif fmt == "tensorboard":
                output_formats.append(TensorBoardOutputFormat(folder))

        self.trained_offset = trained_offset
        # Fixed: `folder` must be passed to Logger
        super().__init__(folder=folder, output_formats=output_formats)

    def record(self, key, value, exclude=(), include=None):
        if key == "time/total_timesteps":
            value += self.trained_offset
        super().record(key, value, exclude, include)


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

    def compile_as_predicted(
        self,
        qc: QuantumCircuit,
        file_name: str,
    ) -> tuple[QuantumCircuit, list[str]]:
        """Compiles a given quantum circuit such that the given figure of merit is maximized by using the respectively trained optimized compiler.

        Arguments:
            qc: The quantum circuit to be compiled or the path to a qasm file containing the quantum circuit.

        Returns:
            A tuple containing the compiled quantum circuit and the compilation information. If compilation fails, False is returned.
        """
        folder = rl.helper.get_path_trained_model()
        model_path = folder / file_name

        #trained_rl_model = rl.helper.load_model("model_" + self.figure_of_merit + "_" + self.device_name)
        trained_rl_model = rl.helper.load_model(str(model_path))

        obs, _ = self.env.reset(qc, seed=0)

        used_compilation_passes = []
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action_masks = get_action_masks(self.env)
            action, _ = trained_rl_model.predict(obs, action_masks=action_masks)
            action = int(action)
            action_item = self.env.action_set[action]
            used_compilation_passes.append(action_item["name"])
            obs, _reward_val, terminated, truncated, _info = self.env.step(action)

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
        custom_callbacks: list[BaseCallback] = None,
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

        if os.path.exists(ckpt_path):
            logger.info(f"📦 Loading checkpoint from {ckpt_path}")
            model = MaskablePPO.load(
                ckpt_path,
                env=self.env,
                tensorboard_log=log_dir,
                verbose=verbose,
                device="cuda",  # or "cpu"
            )
        else:
            logger.info("🆕 No checkpoint found, starting fresh training")
            model = MaskablePPO(
                MaskableMultiInputActorCriticPolicy,
                self.env,
                verbose=verbose,
                tensorboard_log=log_dir,
                gamma=0.98,
                n_steps=n_steps,
            )

        remaining = timesteps - trained

        """ 
            callback = OffsetCheckpointCallback(
            save_freq=n_steps,
            save_path=checkpoint_dir, 
            name_prefix=name_prefix,
            offset=trained,
            verbose=1,
        ) """
        os.makedirs(f"./checkpoints/{save_name}", exist_ok=True)
        """callbacks = []
        if custom_callbacks:
            callbacks.extend(custom_callbacks)
        callback = CallbackList(callbacks) """
        callback = CurriculumProgressionCallback(env=self.env, threshold=0.3, check_freq=50)
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
