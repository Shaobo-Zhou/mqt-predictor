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

# Normalize only num_qubits
scaler = MinMaxScaler()
df["num_qubits_norm"] = scaler.fit_transform(df[["num_qubits"]])

# Use normalized num_qubits and raw normalized-range features for complexity
features_for_complexity = ["num_qubits_norm", "program_communication", "parallelism", "entanglement_ratio", "liveness"]
df["complexity"] = df[features_for_complexity].mean(axis=1)

# Bin into curriculum buckets
df["complexity_bin"] = pd.qcut(df["complexity"], q=5, labels=["very_easy", "easy", "medium", "hard", "very_hard"])

# Save to CSV
output_path = Path(__file__).resolve().parent / "circuit_complexity_metrics.csv"
df.to_csv(output_path, index=False)

# ---- Plotting ----

# Plot histogram of complexity
plt.figure(figsize=(8, 5))
sns.histplot(df["complexity"], bins=20, kde=True, color="skyblue", edgecolor="black")
plt.title("Distribution of Circuit Complexity (Equal Weights)")
plt.xlabel("Complexity Score")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot violin plots
metrics = ["num_qubits", "program_communication", "parallelism", "entanglement_ratio", "liveness"]
fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 6))

for i, metric in enumerate(metrics):
    ax = axes[i]
    sns.violinplot(y=df[metric], ax=ax, inner="box", color="skyblue", cut=0, linewidth=1)
    ax.set_title(metric.replace("_", " ").title())
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    if metric != "num_qubits":
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(-0.2, 1.2)

plt.suptitle("Distribution of Circuit Features", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
