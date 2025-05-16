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

    # rl_pred = Predictor(
    #         figure_of_merit=args.figure_of_merit,
    #         device_name=args.device_name
    #     )
    
    # base_path = rl.helper.get_path_training_circuits() / "training_data_compilation"
    # file_list = list(base_path.rglob("*.qasm"))


    """ results = []
    failed = []


    for idx, file in enumerate(tqdm(file_list, desc="Evaluating circuits"), 1):
        try:
            qc = QuantumCircuit.from_qasm_file(str(file))

            # Pre-filter (optional)
            # if qc.depth() > 15000 or qc.size() > 15000:
            #     print(f"⚠️ Skipping large circuit: {file.name}")
            #     continue

            num_qubits = qc.num_qubits
            depth = qc.depth()
            gate_count = qc.size()

            _, reward_val, compilation_information = rl_pred.compile_as_predicted(qc, model_path)

            results.append({
                "filename": file.name,
                "num_qubits": num_qubits,
                "depth": depth,
                "gate_count": gate_count,
                "reward": reward_val,
                "ep_len": len(compilation_information)
            })

            print(f"[{idx}/{len(file_list)}] ✅ {file.name} | "
                f"Reward: {reward_val:.4f} | Qubits: {num_qubits}, Depth: {depth}, Gates: {gate_count} ")

        except Exception as e:
            failed.append({"filename": file.name, "error": str(e)})
            print(f"[{idx}/{len(file_list)}] ❌ Failed on {file.name}: {e}")

        finally:
            del qc
            gc.collect()

    # --- Save Results ---
    df = pd.DataFrame(results)
    df.to_csv("training_rewards.csv", index=False)
    print("📁 Saved reward data to training_rewards.csv")

    if failed:
        pd.DataFrame(failed).to_csv("failed_circuits.csv", index=False)
        print("📁 Saved failed circuits to failed_circuits.csv")

    # --- Plotting ---
    def plot_metric(metric: str, ylabel: str):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df.sort_values(metric), x=metric, y="reward", marker="o")
        plt.title(f"Reward vs. {ylabel}")
        plt.xlabel(ylabel)
        plt.ylabel("Reward")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"reward_vs_{metric}.png")
        plt.show()

    plot_metric("num_qubits", "Number of Qubits")
    plot_metric("depth", "Circuit Depth")
    plot_metric("gate_count", "Gate Count") """


    """ results_dir = Path(str(resources.files("mqt.predictor"))) / "rl" / "results"
    csv_path = results_dir / "ghz_rewards_new.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results file at {csv_path}")

    df_existing = pd.read_csv(csv_path)

    # --- Configuration: Only evaluate the commented-out models ---
    model_paths = {
        "level_2": Path("curriculum_progression") / "model_level_2",
        "level_3": Path("curriculum_progression") / "model_level_3",
        "model_final": Path("curriculum_progression") / "model_final",
    }
    ghz_range = range(2, 31)
    N = 5  # number of stochastic rollouts

    new_results = []

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

                best_reward = -1
                best_length = 0
                for _ in range(N):
                    _, reward, compilation_information = rl_pred.compile_as_predicted(qc, model_path)
                    if reward > best_reward:
                        best_reward = reward
                        best_length = len(compilation_information)

                new_results.append({
                    "model": model_label,
                    "qubit_size": size,
                    "reward": best_reward,
                    "ep_length": best_length
                })

                print(f"✅ Size {size} | Reward: {best_reward:.4f} | Steps: {best_length}")
            except Exception as e:
                print(f"❌ Failed on GHZ-{size}: {e}")

    # --- Combine and Save ---
    df_combined = pd.concat([df_existing, pd.DataFrame(new_results)], ignore_index=True)
    df_combined.to_csv(csv_path, index=False)
    print(f"\n📁 Appended new results to: {csv_path}") """

    """ test_dir = Path("src/mqt/predictor/rl/training_data/training_circuits/representative_test_data")
    model_paths = {
        #"baseline": Path("bqskit_4") / "rl_expected_fidelity_ibm_washington",
        "model_final": Path("curriculum_progression/model_final"),
        "level_1": Path("curriculum_progression/model_level_2"),
        #"level_2": Path("curriculum_progression/model_level_2"),
        #"level_3": Path("curriculum_progression/model_level_3"),
        "level_4": Path("curriculum_progression/model_level_2"),
        
    }
    results_dir = Path("src/mqt/predictor/rl/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "representative_test_results.csv"

    N=8
    # Evaluate and save after each model
    for model_label, model_path in model_paths.items():
        print(f"\n🚀 Evaluating model: {model_label}")

        rl_pred = Predictor(
            figure_of_merit="expected_fidelity",
            device_name="ibm_washington"
        )

        model_results = []
        for file_path in test_dir.glob("*.qasm"):
            try:
                qc = QuantumCircuit.from_qasm_file(str(file_path))

                best_reward = -1
                best_length = 0
                for _ in range(N):
                    _, reward, compilation_information = rl_pred.compile_as_predicted(qc, model_path)
                    if reward > best_reward:
                        best_reward = reward
                        best_length = len(compilation_information)

                model_results.append({
                    "model": model_label,
                    "file": file_path.name,
                    "qubit_size": qc.num_qubits,
                    "gate_count": qc.size(),
                    "reward": best_reward,
                    "ep_length": best_length
                })

                print(f"✅ Size {qc.num_qubits} | File: {file_path.name} | Reward: {best_reward:.4f} | Steps: {best_length}")

            except Exception as e:
                print(f"❌ Failed on {file_path.name}: {e}")

        # Append to CSV after each model
        df_model = pd.DataFrame(model_results)
        if output_path.exists():
            df_existing = pd.read_csv(output_path)
            df_combined = pd.concat([df_existing, df_model], ignore_index=True)
        else:
            df_combined = df_model

        df_combined.sort_values(by=["gate_count", "model"], inplace=True)
        df_combined.to_csv(output_path, index=False)
        print(f"📁 Results for {model_label} saved to: {output_path}") """

    results_dir = Path("src/mqt/predictor/rl/results")

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
    plt.show()