import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit
from pathlib import Path
from mqt.bench.utils import calc_supermarq_features

def get_path_training_circuits() -> Path:
    return Path(__file__).resolve().parent / "training_data" / "training_circuits"

def load_all_features(max_qubits: int = None):
    base_path = get_path_training_circuits() / "mqt_bench_training"
    file_list = list(base_path.rglob("*.qasm"))

    features = []
    for file in file_list:
        try:
            qc = QuantumCircuit.from_qasm_file(str(file))
            if max_qubits and qc.num_qubits > max_qubits:
                continue
            f = calc_supermarq_features(qc)
            features.append({
                "file": str(file),
                "num_qubits": qc.num_qubits,
                "depth": qc.depth,
                "gate_count": len(qc.data),
                "program_communication": f.program_communication,
                "parallelism": f.parallelism,
                "entanglement_ratio": f.entanglement_ratio,
                "liveness": f.liveness
            })
        except Exception:
            continue
    return pd.DataFrame(features)

# Load features
df = load_all_features()

# Normalize selected features
scaler = MinMaxScaler()
df["num_qubits_norm"] = scaler.fit_transform(df[["num_qubits"]])
df["gate_count_norm"] = scaler.fit_transform(df[["gate_count"]])

# Complexity Metric 1: Full average
features_full = ["num_qubits_norm", "gate_count_norm", "program_communication", "parallelism", "entanglement_ratio", "liveness"]
df["complexity"] = df[features_full].mean(axis=1)

# Complexity Metric 2: Only normalized qubit count
df["complexity_qubits_only"] = df["num_qubits_norm"]

# Complexity Metric 3: Weighted average of normalized qubits and gate count
df["complexity_qubits_gates"] = (df["num_qubits_norm"] + df["gate_count_norm"]) / 2

# Bin each complexity for curriculum stages
df["bin_full"] = pd.qcut(df["complexity"], q=5, labels=["very_easy", "easy", "medium", "hard", "very_hard"])
df["bin_qubits_only"] = pd.qcut(df["complexity_qubits_only"], q=5, labels=["very_easy", "easy", "medium", "hard", "very_hard"])
df["bin_qubits_gates"] = pd.qcut(df["complexity_qubits_gates"], q=5, labels=["very_easy", "easy", "medium", "hard", "very_hard"])

# Save results
output_path = Path(__file__).resolve().parent / "circuit_complexity_metrics.csv"
df.to_csv(output_path, index=False)

# --- Plot each complexity metric ---
def plot_complexity_distribution(col, title):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], bins=20, kde=True, color="skyblue", edgecolor="black")

    # Mark bin edges
    cuts = pd.qcut(df[col], q=5, retbins=True)[1]
    for cut in cuts[1:-1]:  # skip 0 and 1
        plt.axvline(cut, color="red", linestyle="--", linewidth=1)

    plt.title(title)
    plt.xlabel("Complexity Score")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_complexity_distribution("complexity", "Distribution: Full Feature Complexity")
#plot_complexity_distribution("complexity_qubits_only", "Distribution: Qubit-Only Complexity")
#plot_complexity_distribution("complexity_qubits_gates", "Distribution: Qubit + Gate Complexity")


# Plot violin plots
output_path = Path(__file__).resolve().parent / "circuit_complexity_metrics.csv"
df = pd.read_csv(output_path)
metrics = ["num_qubits", "depth", "gate_count", "program_communication", "parallelism", "entanglement_ratio", "liveness"]
fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 6))

for i, metric in enumerate(metrics):
    ax = axes[i]
    sns.violinplot(y=df[metric], ax=ax, inner="box", color="skyblue", cut=0, linewidth=1)
    ax.set_title(metric.replace("_", " ").title())
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    if (metric != "num_qubits") and (metric != "gate_count") and (metric != "depth"):
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(-0.2, 1.2)

plt.suptitle("Distribution of Circuit Features", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
