from mqt.bench import get_benchmark
from predictor import Predictor
from mqt.predictor import reward, rl, qcompile
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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

    rewards = []
    failed = []
    ep_lengths = []

    """ for idx, file in enumerate(file_list, 1):
        try:
            _, reward_val, compilation_information = rl_pred.compile_as_predicted(file, model_path)
            rewards.append(reward_val)
            ep_lengths.append(len(compilation_information))
            
            # Compute current averages
            average_reward = sum(rewards) / len(rewards)
            average_ep_length = sum(ep_lengths) / len(ep_lengths)

            print(f"[{idx}/{len(file_list)}] ✅ {file.name} | Reward: {reward_val:.4f} | "
                f"Avg Reward: {average_reward:.4f} | Avg Ep Len: {average_ep_length:.2f}")
        except Exception as e:
            failed.append((file, str(e)))
            print(f"[{idx}/{len(file_list)}] ❌ Failed on {file.name}: {e}") """
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
    df.to_csv("ghz_rewards.csv", index=False)
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
