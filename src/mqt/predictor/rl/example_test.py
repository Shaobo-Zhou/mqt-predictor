from mqt.bench import get_benchmark
from mqt.predictor import reward, rl
from predictor import Predictor
from mqt.bench.devices import get_device_by_name
from qiskit import QuantumCircuit, transpile
from pytket.extensions.qiskit import qiskit_to_tk, tk_to_qiskit
from pytket.passes import (
    FullPeepholeOptimise, RemoveRedundancies, CliffordSimp, SequencePass, DecomposeBoxes, RoutingPass
)
from pytket.placement import GraphPlacement
from pytket.architecture import Architecture

from pathlib import Path
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
import gc
import psutil, os
from importlib import resources

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL Predictor for Quantum Compilation")
    parser.add_argument("figure_of_merit", type=str, help="The figure of merit to optimize (e.g., expected_fidelity)")
    parser.add_argument("--device_name", type=str, default="ibm_washington", help="Target quantum device")

    args = parser.parse_args()


    test_dir = rl.helper.get_path_training_circuits() / "new_indep_circuits" / "special_test"
    model_dir = rl.helper.get_path_trained_model()
    model_paths = {
        #"baseline": model_dir / "nobqskit" / "rl_expected_fidelity_ibm_washington",
        "baseline_new_data": model_dir / "rl_old_new_data" / "rl_expected_fidelity_ibm_washington",
        #"baseline_0.02": model_dir / "rl_new_reward_0.02" / "rl_expected_fidelity_ibm_washington",
        #"baseline_mqt": model_dir / "mqt_30_new" / "rl_expected_fidelity_ibm_washington",
        #"model_final": model_dir / "curr" / "model_final",
        #"curr_combined": model_dir / "curr_combined" / "model_expected_fidelity_ibm_washington",
        #"level_1": model_dir / "curr" / "model_level_1",
        #"level_2": model_dir / "curr" / "model_final_2",
        #"level_3": model_dir / "curr" / "model_final_3",
        #"level_4": model_dir / "curr" / "model_level_4",
        
    }
    results_dir = Path(__file__).resolve().parent / "results" / "old_new_data"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "special_test.csv"
    #output_path = results_dir / "general_test.csv"
    #N = 10  # Number of evaluation runs
    N=1
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
    print(f"📁 Results saved to: {output_path}")


    """ results_dir = Path("src/mqt/predictor/rl/results")

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