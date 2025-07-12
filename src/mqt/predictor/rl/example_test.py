from mqt.bench import get_benchmark
from mqt.predictor import reward, rl
from predictor import Predictor
from mqt.bench.devices import get_device_by_name
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import PassManager,CouplingMap
from qiskit.passmanager import ConditionalController
from qiskit.passmanager.flow_controllers import DoWhileController
from qiskit.transpiler.passes import (
    Collect2qBlocks,
    ConsolidateBlocks,
    UnitarySynthesis,
    Optimize1qGatesDecomposition,
    CommutativeCancellation,
    GatesInBasis,
    Depth,
    FixedPoint,
    Size,
    MinimumPoint,
)
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from qiskit.transpiler.preset_passmanagers import common
from pytket.passes import (
    FullPeepholeOptimise, RemoveRedundancies, CliffordSimp, SequencePass, DecomposeBoxes, RoutingPass
)
from pytket.placement import GraphPlacement
from pytket.architecture import Architecture

from pathlib import Path
import pandas as pd
import argparse
import random
import matplotlib.pyplot as plt
import seaborn as sns


from tqdm import tqdm
import gc
import psutil, os
from importlib import resources


def make_qiskit_o3_pm(basis_gates, coupling_map):
    """Constructs the QiskitO3 PM with a do-while loop over the minimum-point condition."""
    # the core list of passes
    pm_passes = [
        Collect2qBlocks(),
        ConsolidateBlocks(basis_gates=basis_gates),
        UnitarySynthesis(basis_gates=basis_gates, coupling_map=coupling_map),
        Optimize1qGatesDecomposition(basis=basis_gates),
        CommutativeCancellation(basis_gates=basis_gates),
        GatesInBasis(basis_gates),
        # add a conditional controller around your generated translation sub‐PM
        ConditionalController(
            common.generate_translation_passmanager(
                target=None,
                basis_gates=basis_gates,
                coupling_map=coupling_map
            ).to_flow_controller(),
            condition=lambda ps: not ps["all_gates_in_basis"],
        ),
        Depth(recurse=True),
        FixedPoint("depth"),
        Size(recurse=True),
        FixedPoint("size"),
        MinimumPoint(["depth", "size"], "optimization_loop"),
    ]
    pm = PassManager()
    # Append with a do_while loop: run until the optimization_loop flag is True
    pm.append(
        DoWhileController(
            pm_passes,
            do_while=lambda ps: not ps["optimization_loop_minimum_point"]
        )
    )
    return pm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL Predictor for Quantum Compilation")
    parser.add_argument("figure_of_merit", type=str, help="The figure of merit to optimize (e.g., expected_fidelity)")
    parser.add_argument("--device_name", type=str, default="ibm_washington", help="Target quantum device")

    args = parser.parse_args()

    # rl_pred = Predictor(
    #         figure_of_merit=args.figure_of_merit,
    #         device_name=args.device_name
    #     )
    
    # base_path = rl.helper.get_path_training_circuits() / "training_data_compilation"
    # file_list = list(base_path.rglob("*.qasm"))



    """ test_dir = rl.helper.get_path_training_circuits() / "filtered_and_clustered_circuits" / "test"
    model_dir = rl.helper.get_path_trained_model()
    model_paths = {
        "baseline_new_obs": model_dir / "rl_new_obs_0.01" / "rl_expected_fidelity_ibm_washington",
        #"model_final": model_dir / "curr" / "model_final",
        #"curr": model_dir / "mqt_30" / "rl_expected_fidelity_ibm_washington",
        #"level_1": model_dir / "curr" / "model_level_1",
        #"level_2": model_dir / "curr" / "model_final_2",
        #"level_3": model_dir / "curr" / "model_final_3",
        #"level_4": model_dir / "curr" / "model_level_4",
        
    }
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "baseline_new_obs_0.005_ghz_results.csv"
    ghz_range = range(2, 31)
    N = 1  # number of stochastic rollouts

    model_results = []

    # --- Evaluation Loop ---
    for model_label, model_path in model_paths.items():
        print(f"\n🚀 Evaluating model: {model_label}")

        rl_pred = Predictor(
            figure_of_merit=args.figure_of_merit,
            device_name=args.device_name
        )

        for size in ghz_range:
            try:
                qc = get_benchmark("ghz", level="indep", circuit_size=size)

                rewards = []
                lengths = []
                for _ in range(N):
                    _, reward, compilation_information = rl_pred.compile_as_predicted(qc, model_path)
                    rewards.append(reward)
                    lengths.append(len(compilation_information))

                model_results.append({
                    "model": model_label,
                    "qubit_size": qc.num_qubits,
                    "gate_count": qc.size(),
                    "reward_max": max(rewards),
                    "reward_mean": sum(rewards) / N,
                    "reward_std": pd.Series(rewards).std(),
                    "ep_length_mean": sum(lengths) / N
                })

                print(f"✅ Size {qc.num_qubits} | "
                    f"Reward (max/mean/std): {max(rewards):.4f} / {sum(rewards)/N:.4f} / {pd.Series(rewards).std():.4f} | "
                    f"Mean Steps: {sum(lengths)/N:.1f}")
            except Exception as e:
                print(f"❌ Failed on GHZ-{size}: {e}")

    df_model = pd.DataFrame(model_results)
    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        df_combined = pd.concat([df_existing, df_model], ignore_index=True)
    else:
        df_combined = df_model

    df_combined.sort_values(by=["qubit_size", "model"], inplace=True)
    df_combined.to_csv(output_path, index=False)
    print(f"📁 Results saved to: {output_path}") """

    test_dir = rl.helper.get_path_training_circuits() / "new_indep_circuits" / "special_test"

    device_name = "ibm_washington"
    device = get_device_by_name(device_name)

    N = 20  # Number of transpile runs per circuit for Qiskit

    qiskit_results = []
    for file_path in test_dir.glob("*.qasm"):
        try:
            qc = QuantumCircuit.from_qasm_file(str(file_path))
            fidelities = []
            lengths = []
            for _ in range(N):
                transpiled_qc_qiskit = transpile(
                    qc,
                    basis_gates=device.basis_gates,
                    coupling_map=device.coupling_map,
                    optimization_level=3,
                )
                fidelity = reward.expected_fidelity(transpiled_qc_qiskit, device)
                fidelities.append(fidelity)

            qiskit_results.append({
                "model": "qiskit",
                "file": file_path.name,
                "qubit_size": qc.num_qubits,
                "gate_count": qc.size(),
                "reward_max": max(fidelities),
                "reward_mean": sum(fidelities) / N,
                "reward_std": pd.Series(fidelities).std(),
            })

            print(f"✅ [Qiskit] Size {qc.num_qubits} | File: {file_path.name} | "
                f"Reward (max/mean/std): {max(fidelities):.4f} / {sum(fidelities)/N:.4f} / {pd.Series(fidelities).std():.4f} | "
                f"Mean Steps: {sum(lengths)/N:.1f}")
        except Exception as e:
            print(f"❌ [Qiskit] Failed on {file_path.name}: {e}")
        
    results_dir = Path(__file__).resolve().parent / "results" / "new_data"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "special_test_qiskit.csv"
    df = pd.DataFrame(qiskit_results)
    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
    else:
        df_combined = df

    df_combined.sort_values(by=["gate_count", "model"], inplace=True)
    df_combined.to_csv(output_path, index=False)
    print(f"📁 Results saved to: {output_path}")


    """ test_dir = rl.helper.get_path_training_circuits() / "new_indep_circuits" / "test"
    model_dir = rl.helper.get_path_trained_model()
    model_paths = {
        #"baseline": model_dir / "nobqskit" / "rl_expected_fidelity_ibm_washington",
        "baseline_new_data": model_dir / "rl_new_data_0.005" / "rl_expected_fidelity_ibm_washington",
        #"baseline_0.02": model_dir / "rl_new_reward_0.02" / "rl_expected_fidelity_ibm_washington",
        #"baseline_mqt": model_dir / "mqt_30_new" / "rl_expected_fidelity_ibm_washington",
        #"model_final": model_dir / "curr" / "model_final",
        #"curr_combined": model_dir / "curr_combined" / "model_expected_fidelity_ibm_washington",
        #"level_1": model_dir / "curr" / "model_level_1",
        #"level_2": model_dir / "curr" / "model_final_2",
        #"level_3": model_dir / "curr" / "model_final_3",
        #"level_4": model_dir / "curr" / "model_level_4",
        
    }
    results_dir = Path(__file__).resolve().parent / "results" / "new_data_0.005"
    results_dir.mkdir(parents=True, exist_ok=True)
    #output_path = results_dir / "special_cases.csv"
    output_path = results_dir / "general_test.csv"
    #N = 10  # Number of evaluation runs
    N=20
    model_results = []

    # Evaluate and save after each model
    for model_label, model_path in model_paths.items():
        print(f"\n🚀 Evaluating model: {model_label}")

        rl_pred = Predictor(
            figure_of_merit="expected_fidelity",
            device_name="ibm_washington"
        )

        for file_path in test_dir.glob("*.qasm"):
            file_name = file_path.name
            try:
                qc = QuantumCircuit.from_qasm_file(str(file_path))
                rewards = []
                lengths = []

                for _ in range(N):
                    _, reward, compilation_information = rl_pred.compile_as_predicted(qc, model_path)
                    rewards.append(reward)
                    lengths.append(len(compilation_information))

                model_results.append({
                    "model": model_label,
                    "file": file_path.name,
                    "qubit_size": qc.num_qubits,
                    "gate_count": qc.size(),
                    "reward_max": max(rewards),
                    "reward_mean": sum(rewards) / N,
                    "reward_std": pd.Series(rewards).std(),
                    "ep_length_mean": sum(lengths) / N
                })

                print(f"✅ Size {qc.num_qubits} | File: {file_path.name} | "
                    f"Reward (max/mean/std): {max(rewards):.4f} / {sum(rewards)/N:.4f} / {pd.Series(rewards).std():.4f} | "
                    f"Mean Steps: {sum(lengths)/N:.1f}")

            except Exception as e:
                print(f"❌ Failed on {file_path.name}: {e}")

    # Save results
    df_model = pd.DataFrame(model_results)
    if output_path.exists():
        df_existing = pd.read_csv(output_path)
        df_combined = pd.concat([df_existing, df_model], ignore_index=True)
    else:
        df_combined = df_model
 
    df_combined.sort_values(by=["gate_count", "model"], inplace=True)
    df_combined.to_csv(output_path, index=False)
    print(f"📁 Results saved to: {output_path}") """



    """ results_path = results_dir / "supermarq_test_results.csv"
    df = pd.read_csv(results_path)

    # Filter out rows that need re-evaluation
    needs_rerun = df[df["ep_length_mean"] == 200.0]
    print(f"Will re-run evaluation for {len(needs_rerun)} rows")

    N = 10

    for idx, row in needs_rerun.iterrows():
        model_label = row['model']
        file_name = row['file']
        model_path = model_paths[model_label] if model_label in model_paths else None
        file_path = test_dir / file_name

        if not model_path or not file_path.exists():
            print(f"Skipping: model_path/file_path not found for {model_label} {file_name}")
            continue

        rl_pred = Predictor(
            figure_of_merit="expected_fidelity",
            device_name="ibm_washington"
        )

        rewards = []
        lengths = []
        for _ in range(N):
            try:
                qc = QuantumCircuit.from_qasm_file(str(file_path))
                # file_name as argument for new compile_as_predicted signature
                _, reward, compilation_information = rl_pred.compile_as_predicted(qc, model_path)
                rewards.append(reward)
                lengths.append(len(compilation_information))
            except Exception as e:
                print(f"Failed on {file_name} ({model_label}): {e}")
                rewards.append(0)
                lengths.append(200)  # or whatever sentinel you prefer

        # Update the corresponding row in df
        df.loc[(df['model'] == model_label) & (df['file'] == file_name), [
            'reward_max', 'reward_mean', 'reward_std', 'ep_length_mean'
        ]] = [
            max(rewards),
            sum(rewards) / N,
            pd.Series(rewards).std(),
            sum(lengths) / N
        ]
        print(f"✅ Size {qc.num_qubits} | File: {file_path.name} | "
                    f"Reward (max/mean/std): {max(rewards):.4f} / {sum(rewards)/N:.4f} / {pd.Series(rewards).std():.4f} | "
                    f"Mean Steps: {sum(lengths)/N:.1f}")

    # Save the updated results
    df.to_csv(results_path, index=False)
    print(f"🔄 Results updated in: {results_path}") """

    """results_dir = Path("src/mqt/predictor/rl/results")

    # --- Shared Setup ---
    comparison_pairs = [
        ("baseline", "level_1"),
        ("level_1", "model_final"),
        ("baseline", "model_final"),
    ]
    model_order = ["baseline", "level_1", "model_final"]
    model_palette = dict(zip(model_order, sns.color_palette("colorblind")[:3]))
    linestyles = {
        "baseline": "solid",
        "level_1": "dashed",
        "model_final": "dashed"
    }

    # --- Plot 1: Representative Test Set ---
    rep_path = results_dir / "representative_test_results.csv"
    df_rep = pd.read_csv(rep_path)
    df_rep.sort_values(by=["model", "gate_count"], inplace=True)
    df_rep.to_csv(rep_path, index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (model_a, model_b) in zip(axes, comparison_pairs):
        pair_df = df_rep[df_rep["model"].isin([model_a, model_b])]
        for model in [model_a, model_b]:
            subset = pair_df[pair_df["model"] == model].sort_values("gate_count")
            ax.plot(subset["gate_count"], subset["reward"],
                    marker="o", linestyle=linestyles[model],
                    color=model_palette[model], label=model)
        for gate in sorted(pair_df["gate_count"].unique()):
            best = pair_df[pair_df["gate_count"] == gate].sort_values("reward", ascending=False).iloc[0]
            ax.plot(best["gate_count"], best["reward"], marker="*", color=model_palette[best["model"]], markersize=12)
        ax.set_title(f"{model_a} vs {model_b}")
        ax.set_xlabel("Gate Count")
        ax.grid(True)
        ax.legend(title="Model", loc="upper right")
    axes[0].set_ylabel("Reward")
    plt.suptitle("Model Comparison on Representative Test Set", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(results_dir / "representative_testset_comparison_subplots.png")
    plt.show()

    # --- Plot 2: GHZ Evaluation ---
    ghz_path = results_dir / "ghz_rewards_new.csv"
    df_ghz = pd.read_csv(ghz_path)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for ax, (model_a, model_b) in zip(axes, comparison_pairs):
        pair_df = df_ghz[df_ghz["model"].isin([model_a, model_b])]
        for model in [model_a, model_b]:
            subset = pair_df[pair_df["model"] == model].sort_values("qubit_size")
            ax.plot(subset["qubit_size"], subset["reward"],
                    marker="o", linestyle=linestyles[model],
                    color=model_palette[model], label=model)
        for q in sorted(pair_df["qubit_size"].unique()):
            best = pair_df[pair_df["qubit_size"] == q].sort_values("reward", ascending=False).iloc[0]
            ax.plot(best["qubit_size"], best["reward"], marker="*", color=model_palette[best["model"]], markersize=12)
        ax.set_title(f"{model_a} vs {model_b}")
        ax.set_xlabel("GHZ Circuit Size (Qubits)")
        ax.legend(title="Model", loc="upper right")
        ax.grid(True)
    axes[0].set_ylabel("Reward")
    plt.suptitle("GHZ Reward Comparison Across Models", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(results_dir / "ghz_rewards_comparison_subplots.png")
    plt.show() """