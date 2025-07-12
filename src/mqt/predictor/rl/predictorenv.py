"""Predictor environment for the compilation using reinforcement learning."""

from __future__ import annotations

import logging
import copy
import sys
import gc
import time
import json
import warnings
from typing import TYPE_CHECKING, Any

if sys.version_info >= (3, 11) and TYPE_CHECKING:  # pragma: no cover
    from typing import assert_never
else:
    from typing_extensions import assert_never

if TYPE_CHECKING:
    from pathlib import Path

import random
import numpy as np
import pandas as pd
from bqskit.ext import bqskit_to_qiskit, qiskit_to_bqskit
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete
from joblib import load
from pytket.circuit import Qubit
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from qiskit import QuantumCircuit
from qiskit.passmanager.flow_controllers import DoWhileController
from qiskit.transpiler import CouplingMap, PassManager, TranspileLayout
from qiskit.transpiler.passes import CheckMap, GatesInBasis
from qiskit.transpiler.passes.layout.vf2_layout import VF2LayoutStopReason
from qiskit.circuit.library.standard_gates import RCCXGate


from mqt.bench.devices import get_device_by_name
from mqt.predictor import reward, rl
from mqt.predictor.hellinger import get_hellinger_model_path

logger = logging.getLogger("mqt-predictor")
logger.propagate = False


class PredictorEnv(Env):  # type: ignore[misc]
    """Predictor environment for reinforcement learning."""

    def __init__(
        self, reward_function: reward.figure_of_merit = "expected_fidelity", device_name: str = "ibm_washington",
        reward_neg: float = -0.01,
        reward_pos: float = 1.0,
    ) -> None:
        """Initializes the PredictorEnv object."""
        logger.info("Init env: " + reward_function)

        self.action_set = {}
        self.actions_synthesis_indices = []
        self.actions_layout_indices = []
        self.actions_routing_indices = []
        self.actions_mapping_indices = []
        self.actions_opt_indices = []
        self.actions_final_optimization_indices = []
        self.used_actions: list[str] = []
        self.action_timings = {}
        self.device = get_device_by_name(device_name)
        self.curriculum_df = None
        self.curriculum_bins = None
        self.current_difficulty_level = 0
        self.max_difficulty_level = 4  
        self.curriculum_sampling_enabled = False
        # check for uni-directional coupling map
        for a, b in self.device.coupling_map:
            if [b, a] not in self.device.coupling_map:
                msg = f"The connectivity of the device '{device_name}' is uni-directional and MQT Predictor might return a compiled circuit that assumes bi-directionality."
                warnings.warn(msg, UserWarning, stacklevel=2)

        index = 0

        for elem in rl.helper.get_actions_synthesis():
            self.action_set[index] = elem
            self.actions_synthesis_indices.append(index)
            index += 1
        for elem in rl.helper.get_actions_layout():
            self.action_set[index] = elem
            self.actions_layout_indices.append(index)
            index += 1
        for elem in rl.helper.get_actions_routing():
            self.action_set[index] = elem
            self.actions_routing_indices.append(index)
            index += 1
        for elem in rl.helper.get_actions_opt():
            self.action_set[index] = elem
            self.actions_opt_indices.append(index)
            index += 1
        for elem in rl.helper.get_actions_mapping():
            self.action_set[index] = elem
            self.actions_mapping_indices.append(index)
            index += 1
        for elem in rl.helper.get_actions_final_optimization():
            self.action_set[index] = elem
            self.actions_final_optimization_indices.append(index)
            index += 1

        self.action_set[index] = rl.helper.get_action_terminate()
        self.action_terminate_index = index

        if reward_function == "estimated_success_probability" and not reward.esp_data_available(self.device):
            msg = f"Missing calibration data for ESP calculation on {device_name}."
            raise ValueError(msg)
        if reward_function == "estimated_hellinger_distance":
            hellinger_model_path = get_hellinger_model_path(self.device)
            if not hellinger_model_path.is_file():
                msg = f"Missing trained model for Hellinger distance estimates on {self.device.name}."
                raise ValueError(msg)
            self.hellinger_model = load(hellinger_model_path)
        self.reward_function = reward_function
        self.action_space = Discrete(len(self.action_set.keys()))
        self.num_steps = 0
        self.layout: TranspileLayout | None = None
        self.num_qubits_uncompiled_circuit = 0

        self.has_parameterized_gates = False

        spaces = {}
        spaces = {
            "num_qubits": Discrete(128),
            "depth": Discrete(75000),
            #"gate_count": Discrete(100000),
            "program_communication": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "critical_depth": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "entanglement_ratio": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "parallelism": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "liveness": Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "only_nat_gates": Discrete(2),
            "mapped": Discrete(2)
        }
        for gate in rl.helper.get_openqasm_gates():
            if gate in ['rz', 'sx', 'x', 'cx', 'y', 'z', 'h', 'swap','s', 'sdg', 'u']:
                spaces[gate] = Discrete(120000)
            else:
                spaces[gate] = Discrete(6000)
        spaces["native_mask"] = Box(low=0, high=1, shape=(len(rl.helper.get_openqasm_gates()),), dtype=np.int32)
        
        
        spaces["measure"] = Discrete(128)
        # spaces["avg_hopcount"] = Box(low=-1, high=128, shape=(1,), dtype=np.float32)  
        # spaces["max_degree"] = Discrete(128)
        # spaces["min_degree"] = Discrete(128)
        #spaces["adj_std"] = Box(low=-1, high=10000, shape=(1,), dtype=np.float32)
        self.observation_space = Dict(spaces)
        self.filename = ""
        self.action_effectiveness = {
            self.action_set[i]["name"]: {"changed": 0, "unchanged": 0}
            for i in self.action_set
        }
        self.reward_neg = reward_neg
        self.reward_pos = reward_pos


    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[Any, Any]]:
        """Executes the given action and returns the new state, the reward, whether the episode is done, whether the episode is truncated and additional information."""
        action_name = self.action_set[action].get("name")
        logger.info(f"🛠️  [Step {self.num_steps}] Applying action: {action_name}")
        self.used_actions.append(str(action_name))

        prev_obs = rl.helper.create_feature_dict(self.state, self.device.basis_gates, self.device.coupling_map)
        prev_fidelity = None

        if self.action_terminate_index in self.determine_valid_actions_for_state():
            try:
                prev_fidelity = self.calculate_reward()
            except Exception:
                prev_fidelity = None
        start_time = time.time()
        altered_qc = self.apply_action(action)
        elapsed = time.time() - start_time

        logger.info(f"⏱️  [Step {self.num_steps}] Action '{action_name}' took {elapsed:.2f} seconds")

        if not altered_qc:
            gc.collect()
            return (
                rl.helper.create_feature_dict(self.state, self.device.basis_gates, self.device.coupling_map),
                0,
                True,
                False,
                {},
            )

        self.state: QuantumCircuit = altered_qc
        self.num_steps += 1

        self.valid_actions = self.determine_valid_actions_for_state()
        if len(self.valid_actions) == 0:
            msg = "No valid actions left."
            raise RuntimeError(msg)

        if action == self.action_terminate_index:
            reward_val = self.calculate_reward()
            logger.info(f"Final Fidelity: {reward_val}")
            done = True
        else:
            reward_val = self.reward_neg
            new_obs = rl.helper.create_feature_dict(self.state, self.device.basis_gates, self.device.coupling_map)
            if self.features_changed(prev_obs, new_obs):
                self.action_effectiveness[action_name]["changed"] += 1
                #reward_val = 0
                if self.action_terminate_index in self.valid_actions:
                    fidelity_reward = 0
                    try:
                        new_fidelity = self.calculate_reward()
                        if prev_fidelity is not None and new_fidelity > prev_fidelity:
                            reward_val = 0
                            logger.info(f"Fidelity gain of {new_fidelity - prev_fidelity}")
                            # Reward proportional to improvement, or just self.reward_positive_fidelity
                            fidelity_reward = self.reward_pos * (new_fidelity - prev_fidelity)
                        elif prev_fidelity is not None and new_fidelity < prev_fidelity:
                            logger.info(f"Fidelity loss of {new_fidelity - prev_fidelity}")
                            fidelity_reward = self.reward_pos * (new_fidelity - prev_fidelity)
                    except Exception:
                        fidelity_reward = 0
                    reward_val += fidelity_reward
            else:
                self.action_effectiveness[action_name]["unchanged"] += 1
                logger.info("Ineffective action")
                #reward_val = self.reward_neg
                #reward_val = 0

            done = False

        # in case the Qiskit.QuantumCircuit has unitary or u gates in it, decompose them (because otherwise qiskit will throw an error when applying the BasisTranslator
        if self.state.count_ops().get("unitary"):
            self.state = self.state.decompose(gates_to_decompose="unitary")
        elif self.state.count_ops().get("clifford"):
            self.state = self.state.decompose(gates_to_decompose="clifford")

        self.state._layout = self.layout  # noqa: SLF001
        obs = rl.helper.create_feature_dict(self.state, self.device.basis_gates, self.device.coupling_map)
        #print(obs)
        del altered_qc
        gc.collect()
        return obs, reward_val, done, False, {}

    def export_action_timings(self, filepath: str = "action_timings.json"):
        """Export average action timings to a JSON file."""
        avg_timings = {
            action: sum(times) / len(times)
            for action, times in self.action_timings.items()
        }
        with open(filepath, "w") as f:
            json.dump(avg_timings, f, indent=4)
        logger.info(f"📁 Saved average action timings to {filepath}")
    def calculate_reward(self) -> float:
        """Calculates and returns the reward for the current state."""
        if self.reward_function == "expected_fidelity":
            return reward.expected_fidelity(self.state, self.device)
        if self.reward_function == "estimated_success_probability":
            return reward.estimated_success_probability(self.state, self.device)
        if self.reward_function == "estimated_hellinger_distance":
            return reward.estimated_hellinger_distance(self.state, self.device, self.hellinger_model)
        if self.reward_function == "critical_depth":
            return reward.crit_depth(self.state)
        assert_never(self.state)

    def render(self) -> None:
        """Renders the current state."""
        print(self.state.draw())

    def set_curriculum_data(self, df: pd.DataFrame, enable_sampling: bool = True):
        """Load the curriculum dataframe and enable/disable sampling."""
        self.curriculum_df = df
        bin_order = ["very_easy", "easy", "medium", "hard", "very_hard"]
        self.curriculum_bins = [b for b in bin_order if b in df["complexity_bin"].unique()]
        self.curriculum_sampling_enabled = enable_sampling
        self.current_difficulty_level = 0

    def increase_curriculum_difficulty(self) -> bool:
        """Increase difficulty level if possible, return True if updated."""
        if not self.curriculum_sampling_enabled:
            return False
        if self.current_difficulty_level < (self.max_difficulty_level): ### Ignore extreme cases
            self.current_difficulty_level += 1
            logger.info(f"📈 Difficulty increased to level {self.current_difficulty_level}")
            return True
        logger.info("🔒 Already at maximum difficulty level.")
        return False
    def reset(
        self,
        qc: Path | str | QuantumCircuit | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,  # noqa: ARG002
    ) -> tuple[QuantumCircuit, dict[str, Any]]:
        """Resets the environment to the given state or a random state.

        Arguments:
            qc: The quantum circuit to be compiled or the path to a qasm file containing the quantum circuit. Defaults to None.
            seed: The seed to be used for the random number generator. Defaults to None.
            options: Additional options. Defaults to None.

        Returns:
            The initial state and additional information.
        """
        super().reset(seed=seed)
        if self.curriculum_sampling_enabled and self.curriculum_df is not None:
            level_label = self.curriculum_bins[self.current_difficulty_level]
            df_filtered = self.curriculum_df[self.curriculum_df["complexity_bin"] == level_label]
            
            logger.info(f"📚 Curriculum sampling enabled. Current difficulty level: {self.current_difficulty_level} ({level_label})")
            logger.info(f"🧪 Sampling from {len(df_filtered)} circuits at this level.")
            
            if df_filtered.empty:
                raise ValueError(f"No circuits available for difficulty level '{level_label}'.")

            sampled_file = random.choice(df_filtered["file"].tolist())
            logger.info(f"🎯 Sampled circuit: {sampled_file}")
            self.state = QuantumCircuit.from_qasm_file(sampled_file)
            self.filename = sampled_file

        elif isinstance(qc, QuantumCircuit):
            self.state = qc
        elif qc:
            self.state = QuantumCircuit.from_qasm_file(str(qc))
        else:
            rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
            self.state, self.filename = rl.helper.get_state_sample(self.device.num_qubits, rng)
            logger.info(f"🎯 Sampled circuit: {self.filename}")

        self.action_space = Discrete(len(self.action_set.keys()))
        self.num_steps = 0
        self.used_actions = []

        self.layout = None

        self.valid_actions = self.actions_opt_indices + self.actions_synthesis_indices

        self.error_occurred = False

        self.num_qubits_uncompiled_circuit = self.state.num_qubits
        self.has_parameterized_gates = len(self.state.parameters) > 0


        return rl.helper.create_feature_dict(self.state, self.device.basis_gates, self.device.coupling_map), {}

    def action_masks(self) -> list[bool]:
        """Returns a list of valid actions for the current state."""
        action_mask = [action in self.valid_actions for action in self.action_set]

        # it is not clear how tket will handle the layout, so we remove all actions that are from "origin"=="tket" if a layout is set
        if self.layout is not None:
            action_mask = [
                action_mask[i] and self.action_set[i].get("origin") != "tket" for i in range(len(action_mask))
            ]

        if self.has_parameterized_gates or self.layout is not None:
            # remove all actions that are from "origin"=="bqskit" because they are not supported for parameterized gates
            # or after layout since using BQSKit after a layout is set may result in an error
            action_mask = [
                action_mask[i] and self.action_set[i].get("origin") != "bqskit" for i in range(len(action_mask))
            ]

        # only allow VF2PostLayout if "ibm" is in the device name
        if "ibm" not in self.device.name:
            action_mask = [
                action_mask[i] and self.action_set[i].get("name") != "VF2PostLayout" for i in range(len(action_mask))
            ]
        return action_mask

    def apply_action(self, action_index: int) -> QuantumCircuit | None:
        """Applies the given action to the current state and returns the altered state."""
        if action_index not in self.action_set:
            raise ValueError(f"Action {action_index} not supported.")

        action = self.action_set[action_index]
        altered_qc = None
        prop = None  # property_set for Qiskit actions

        # Terminate action (no modification)
        if action["name"] == "terminate":
            return self.state

        try:
            if action["origin"] in {"qiskit", "qiskit_ai"}:
                # --------- QISKIT & QISKIT_AI PATH ---------

                # Remove cregs if AIRouting is in the action name (works for composites too!)
                if "AIRouting" in action["name"]:
                    self.state = self.remove_cregs(self.state)

                if action.get("stochastic", False):
                    # Composite/stochastic action: use best-of-N wrapper
                    altered_qc, prop = rl.helper.best_of_n_passmanager(
                        action["transpile_pass"], self.device, self.state, n_attempts=10
                    )
                else:
                    # Deterministic actions and special handling for some passes
                    if action["name"] == "QiskitO3":
                        pm = PassManager()
                        pm.append(
                            DoWhileController(
                                action["transpile_pass"](
                                    self.device.basis_gates,
                                    CouplingMap(self.device.coupling_map) if self.layout is not None else None,
                                ),
                                do_while=action["do_while"],
                            ),
                        )
                    elif action["name"] in ["Opt2qBlocks", "Optimize1qGatesDecomposition"]:
                        pm = PassManager(action["transpile_pass"](self.device.basis_gates))
                    elif action["name"] == "AIClifford":
                        pm = PassManager(action["transpile_pass"](CouplingMap(self.device.coupling_map)))
                    else:
                        transpile_pass = (
                            action["transpile_pass"](self.device)
                            if callable(action["transpile_pass"]) else action["transpile_pass"]
                        )
                        pm = PassManager(transpile_pass)

                    altered_qc = pm.run(self.state)
                    prop = pm.property_set

                # ---------- PROPERTY SET & LAYOUT HANDLING ----------
                if action_index in (
                    self.actions_layout_indices
                    + self.actions_mapping_indices
                    + self.actions_final_optimization_indices
                ):
                    if action["name"] == "VF2PostLayout":
                        assert prop["VF2PostLayout_stop_reason"] is not None
                        post_layout = prop["post_layout"]
                        if post_layout:
                            altered_qc, _ = rl.helper.postprocess_vf2postlayout(altered_qc, post_layout, self.layout)
                    elif action["name"] == "VF2Layout":
                        if prop["VF2Layout_stop_reason"] == VF2LayoutStopReason.SOLUTION_FOUND:
                            assert prop["layout"]
                    else:
                        assert prop["layout"]

                    if prop.get("layout"):
                        self.layout = TranspileLayout(
                            initial_layout=prop["layout"],
                            input_qubit_mapping=prop["original_qubit_indices"],
                            final_layout=prop["final_layout"],
                            _output_qubit_list=altered_qc.qubits,
                            _input_qubit_count=self.num_qubits_uncompiled_circuit,
                        )

                elif action_index in self.actions_routing_indices:
                    assert self.layout is not None
                    self.layout.final_layout = prop.get("final_layout")

            elif action["origin"] == "tket":
                # -------------- TKET PATH --------------
                try:
                    if any(isinstance(gate[0], RCCXGate) for gate in self.state.data):
                        self.state = self.state.decompose()
                    tket_qc = qiskit_to_tk(self.state, preserve_param_uuid=True)
                    transpile_pass = (
                        action["transpile_pass"](self.device)
                        if callable(action["transpile_pass"]) else action["transpile_pass"]
                    )
                    for elem in transpile_pass:
                        elem.apply(tket_qc)
                    qbs = tket_qc.qubits
                    qubit_map = {qbs[i]: Qubit("q", i) for i in range(len(qbs))}
                    tket_qc.rename_units(qubit_map)
                    altered_qc = tk_to_qiskit(tket_qc, replace_implicit_swaps=True)
                    if action_index in self.actions_routing_indices:
                        assert self.layout is not None
                        self.layout.final_layout = rl.helper.final_layout_pytket_to_qiskit(tket_qc, altered_qc)
                except Exception:
                    logger.exception(
                        f"Error in executing TKET transpile pass for {action['name']} at step {self.num_steps} for {self.filename}"
                    )
                    self.error_occurred = True
                    return None

            elif action["origin"] == "bqskit":
                # ------------- BQSKit PATH --------------
                try:
                    bqskit_qc = qiskit_to_bqskit(self.state)
                    transpile_pass = (
                        action["transpile_pass"](self.device)
                        if callable(action["transpile_pass"]) else action["transpile_pass"]
                    )
                    if action_index in self.actions_opt_indices + self.actions_synthesis_indices:
                        bqskit_compiled_qc = transpile_pass(bqskit_qc)
                        altered_qc = bqskit_to_qiskit(bqskit_compiled_qc)
                    elif action_index in self.actions_mapping_indices:
                        bqskit_compiled_qc, initial_layout, final_layout = transpile_pass(bqskit_qc)
                        altered_qc = bqskit_to_qiskit(bqskit_compiled_qc)
                        layout = rl.helper.final_layout_bqskit_to_qiskit(
                            initial_layout, final_layout, altered_qc, self.state
                        )
                        self.layout = layout
                except Exception:
                    logger.exception(
                        f"Error in executing BQSKit transpile pass for {action['name']} at step {self.num_steps} for {self.filename}"
                    )
                    self.error_occurred = True
                    return None

            else:
                raise ValueError(f"Origin {action['origin']} not supported.")

        except Exception:
            logger.exception(
                f"Error in executing transpile pass for {action['name']} at step {self.num_steps} for {self.filename}"
            )
            self.error_occurred = True
            return None

        return altered_qc
    """ def apply_action(self, action_index: int) -> QuantumCircuit | None:
        #Applies the given action to the current state and returns the altered state.
        if action_index in self.action_set:
            action = self.action_set[action_index]
            if action["name"] == "terminate":
                return self.state
            if action_index in self.actions_opt_indices:
                transpile_pass = action["transpile_pass"]
                if callable(transpile_pass) and action["name"] not in {"QiskitO3", "Opt2qBlocks", "Optimize1qGatesDecomposition", "BQSKitO2", "AIClifford"}:
                    transpile_pass = transpile_pass(self.device)
            else:
                transpile_pass = action["transpile_pass"](self.device)

            if action["origin"] in {"qiskit", "qiskit_ai"}:
                try:
                    if action["name"] == "QiskitO3":
                        pm = PassManager()
                        pm.append(
                            DoWhileController(
                                action["transpile_pass"](
                                    self.device.basis_gates,
                                    CouplingMap(self.device.coupling_map) if self.layout is not None else None,
                                ),
                                do_while=action["do_while"],
                            ),
                        )
                    elif action["name"] in ["Opt2qBlocks", "Optimize1qGatesDecomposition"]:
                        pm = PassManager(
                            action["transpile_pass"](self.device.basis_gates)
                        )
                    elif action["name"] == "AIClifford":
                        pm = PassManager(
                            action["transpile_pass"](CouplingMap(self.device.coupling_map))
                        )
                    else:
                        pm = PassManager(transpile_pass)  
                    if action["name"] == "AIRouting":
                        self.state = self.remove_cregs(self.state)
                    altered_qc = pm.run(self.state)
                    
                except Exception:
                    logger.exception(
                        "Error in executing Qiskit transpile pass for {action} at step {i} for {filename}".format(
                            action=action["name"], i=self.num_steps, filename=self.filename
                        )
                    )

                    self.error_occurred = True
                    return None
                if (
                    action_index
                    in self.actions_layout_indices
                    + self.actions_mapping_indices
                    + self.actions_final_optimization_indices
                ):
                    if action["name"] == "VF2PostLayout":
                        assert pm.property_set["VF2PostLayout_stop_reason"] is not None
                        post_layout = pm.property_set["post_layout"]
                        if post_layout:
                            altered_qc, pm = rl.helper.postprocess_vf2postlayout(altered_qc, post_layout, self.layout)
                    elif action["name"] == "VF2Layout":
                        if pm.property_set["VF2Layout_stop_reason"] == VF2LayoutStopReason.SOLUTION_FOUND:
                            assert pm.property_set["layout"]
                    else:
                        assert pm.property_set["layout"]

                    if pm.property_set["layout"]:
                        self.layout = TranspileLayout(
                            initial_layout=pm.property_set["layout"],
                            input_qubit_mapping=pm.property_set["original_qubit_indices"],
                            final_layout=pm.property_set["final_layout"],
                            _output_qubit_list=altered_qc.qubits,
                            _input_qubit_count=self.num_qubits_uncompiled_circuit,
                        )

                elif action_index in self.actions_routing_indices:
                    assert self.layout is not None
                    self.layout.final_layout = pm.property_set["final_layout"]

            elif action["origin"] == "tket":
                try:
                    if any(isinstance(gate[0], RCCXGate) for gate in self.state.data):
                        self.state = self.state.decompose()
                    tket_qc = qiskit_to_tk(self.state, preserve_param_uuid=True)
                    for elem in transpile_pass:
                        elem.apply(tket_qc)
                    qbs = tket_qc.qubits
                    qubit_map = {qbs[i]: Qubit("q", i) for i in range(len(qbs))}
                    tket_qc.rename_units(qubit_map)  # type: ignore[arg-type]
                    #altered_qc = tk_to_qiskit(tket_qc)
                    altered_qc = tk_to_qiskit(tket_qc, replace_implicit_swaps=True)
                    if action_index in self.actions_routing_indices:
                        assert self.layout is not None
                        self.layout.final_layout = rl.helper.final_layout_pytket_to_qiskit(tket_qc, altered_qc)

                except Exception:
                    logger.exception(
                        "Error in executing TKET transpile  pass for {action} at step {i} for {filename}".format(
                            action=action["name"], i=self.num_steps, filename=self.filename
                        )
                    )
                    self.error_occurred = True
                    return None

            elif action["origin"] == "bqskit":
                try:
                    bqskit_qc = qiskit_to_bqskit(self.state)
                    if action_index in self.actions_opt_indices + self.actions_synthesis_indices:
                        bqskit_compiled_qc = transpile_pass(bqskit_qc)
                        altered_qc = bqskit_to_qiskit(bqskit_compiled_qc)
                    elif action_index in self.actions_mapping_indices:
                        bqskit_compiled_qc, initial_layout, final_layout = transpile_pass(bqskit_qc)
                        altered_qc = bqskit_to_qiskit(bqskit_compiled_qc)
                        layout = rl.helper.final_layout_bqskit_to_qiskit(
                            initial_layout, final_layout, altered_qc, self.state
                        )
                        self.layout = layout
                except Exception:
                    logger.exception(
                        "Error in executing BQSKit transpile pass for {action} at step {i} for {filename}".format(
                            action=action["name"], i=self.num_steps, filename=self.filename
                        )
                    )
                    self.error_occurred = True
                    return None

            else:
                error_msg = f"Origin {action['origin']} not supported."
                raise ValueError(error_msg)

        else:
            error_msg = f"Action {action_index} not supported."
            raise ValueError(error_msg)

        return altered_qc """

    def determine_valid_actions_for_state(self) -> list[int]:
        """Determines and returns the valid actions for the current state."""
        check_nat_gates = GatesInBasis(basis_gates=self.device.basis_gates)
        check_nat_gates(self.state)
        only_nat_gates = check_nat_gates.property_set["all_gates_in_basis"]

        if not only_nat_gates:
            actions = self.actions_synthesis_indices + self.actions_opt_indices
            if self.layout is not None:
                actions += self.actions_routing_indices
            return actions

        check_mapping = CheckMap(coupling_map=CouplingMap(self.device.coupling_map))
        check_mapping(self.state)
        mapped = check_mapping.property_set["is_swap_mapped"]

        if mapped and self.layout is not None:  # The circuit is correctly mapped.
            if self.num_steps > 50:
                return [self.action_terminate_index]
            return [self.action_terminate_index, *self.actions_opt_indices, *self.actions_final_optimization_indices]

        if self.layout is not None:  # The circuit is not yet mapped but a layout is set.
            return self.actions_routing_indices

        # No layout applied yet
        return self.actions_mapping_indices + self.actions_layout_indices + self.actions_opt_indices
    
    def export_action_effectiveness(self, path="action_effectiveness.json"):
        """Export the success/failure count of actions to a JSON file."""
        with open(path, "w") as f:
            json.dump(self.action_effectiveness, f, indent=4)
        logger.info(f"📊 Saved action effectiveness to {path}")
    
    def features_changed(self, prev: dict, curr: dict) -> bool:
        for key in prev:
            if isinstance(prev[key], np.ndarray):
                if not np.allclose(prev[key], curr[key]):
                    return True
            else:
                if prev[key] != curr[key]:
                    return True
        return False
    def remove_cregs(self, qc: QuantumCircuit) -> QuantumCircuit:
        new_qc = QuantumCircuit(qc.num_qubits, name=qc.name)
    
        for instr, qargs, cargs in qc.data:
            # If there are no classical arguments, keep the instruction
            if not cargs:
                new_qc.append(instr, qargs)
            # (Optionally: if you want to keep barriers or certain instrs that don't use cregs, adjust logic here)

        return new_qc






