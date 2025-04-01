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
                "critical_depth": f.critical_depth,
                "parallelism": f.parallelism,
                "entanglement_ratio": f.entanglement_ratio,
            })
        except Exception:
            continue
    return pd.DataFrame(features)


""" # Load feature data
df = load_all_features()

df_outliers = df[
    (df["critical_depth"] > 1) | (df["critical_depth"] < 0) |
    (df["entanglement_ratio"] > 1) | (df["entanglement_ratio"] < 0)
]
print("Outliers detected:", len(df_outliers))
print(df_outliers)


# Features to include in complexity
feature_columns = ["num_qubits", "critical_depth", "parallelism", "entanglement_ratio"]

# Normalize the features
scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df[feature_columns]), columns=feature_columns)

# Compute complexity score as the mean of normalized features (equal weights)
df["complexity"] = df_normalized.mean(axis=1)

df["complexity_bin"] = pd.qcut(df["complexity"], q=5, labels=["very_easy", "easy", "medium", "hard", "very_hard"])

# Save to CSV for reuse
output_path = Path(__file__).resolve().parent / "circuit_complexity_metrics.csv"
df.to_csv(output_path, index=False)

# Plot the complexity distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["complexity"], bins=20, kde=True, color="skyblue", edgecolor="black")
plt.title("Distribution of Circuit Complexity (Equal Weights)")
plt.xlabel("Complexity Score")
plt.ylabel("Count")
plt.grid(True)
plt.tight_layout()
plt.show()
 """

SCRIPT_DIR = Path(__file__).resolve().parent
csv_path = SCRIPT_DIR / "circuit_complexity_metrics.csv"
df = pd.read_csv(csv_path)

# Plot violin plots for the core features + complexity
metrics = ["num_qubits", "critical_depth", "parallelism", "entanglement_ratio"]


# Set up figure and axes
fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 6), sharex=False)
for i, metric in enumerate(metrics):
    ax = axes[i]
    sns.violinplot(y=df[metric], ax=ax, inner="box", color="skyblue", cut=0, linewidth=1)
    
    ax.set_title(metric.replace("_", " ").title())
    ax.set_xlabel("")  # No x-label needed
    ax.set_xticks([])  # Remove x-ticks for cleaner look
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    
    # Add reference lines for normalized metrics
    if metric != "num_qubits":
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(-0.2, 1.2)

# Adjust layout
plt.suptitle("Distribution of Circuit Features", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()