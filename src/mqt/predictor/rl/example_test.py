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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RL Predictor for Quantum Compilation")
    parser.add_argument("figure_of_merit", type=str, help="The figure of merit to optimize (e.g., expected_fidelity)")
    parser.add_argument("--device_name", type=str, default="ibm_washington", help="Target quantum device")

    args = parser.parse_args()

    #qc = get_benchmark("ghz", level="indep", circuit_size=5)
    #qc.draw(output='mpl')

    rl_pred = Predictor(
            figure_of_merit=args.figure_of_merit,
            device_name=args.device_name
        )
    
    base_path = rl.helper.get_path_training_circuits() / "training_data_compilation"
    file_list = list(base_path.rglob("*.qasm"))
    model_path = Path("bqskit_4") / "rl_expected_fidelity_ibm_washington"
    #model_path = Path("nobqskit") / "rl_expected_fidelity_ibm_washington"


    results = []
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
    plot_metric("gate_count", "Gate Count")


    """ rewards = []
    failed = []
    ep_lengths = []

    for i in range(2, 31):
        try:
            qc = get_benchmark("ghz", level="indep", circuit_size=i)
            _, reward_val, compilation_information = rl_pred.compile_as_predicted(qc, model_path)
            
            rewards.append(reward_val)
            ep_lengths.append(len(compilation_information))

            average_reward = sum(rewards) / len(rewards)
            average_ep_length = sum(ep_lengths) / len(ep_lengths)

            print("=" * 60)
            print(f"✅ GHZ circuit size: {i}")
            print(f"🎯 Reward: {reward_val:.4f}")
            print(f"🔁 Episode length: {len(compilation_information)} steps")
            print(f"📊 Running average reward: {average_reward:.4f}")
            print(f"📏 Running average episode length: {average_ep_length:.2f}")
            print("=" * 60)

        except Exception as e:
            failed.append((i, str(e)))
            print("❌ Failed to compile GHZ circuit of size", i)
            print("   Reason:", str(e))


    average_reward = sum(rewards) / len(rewards) if rewards else 0
    average_ep_length = sum(ep_lengths) / len(ep_lengths) if ep_lengths else 0
    print(f"✅ Evaluated {len(rewards)} circuits successfully.")
    print(f"❌ Failed on {len(failed)} circuits.")
    print(f"🎯 Average reward: {average_reward:.4f}")
    print(f"🎯 Average episode length: {average_ep_length:.4f}")

    # Create a DataFrame with qubit size and reward
    df = pd.DataFrame({
        "qubit_size": list(range(2, 2 + len(rewards))),
        "reward": rewards
    })

    # Save to CSV
    #df.to_csv("ghz_rewards.csv", index=False)
    print("📁 Saved rewards to ghz_rewards.csv")
    circuit_sizes = list(range(2, 2 + len(rewards)))  # Assuming rewards list is aligned with qubit sizes

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=circuit_sizes, y=rewards, marker='o')
    plt.title("Reward vs. GHZ Circuit Size")
    plt.xlabel("Number of Qubits (GHZ Circuit Size)")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
 """