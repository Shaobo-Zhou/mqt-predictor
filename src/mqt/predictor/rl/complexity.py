import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from qiskit import QuantumCircuit
from pathlib import Path
from mqt.bench.utils import calc_supermarq_features

def get_path_training_circuits() -> Path:
    return Path(__file__).resolve().parent / "training_data" / "training_circuits"

def load_all_features(max_qubits: int = None, remove_depth_outliers: bool = True, outlier_quantile: float = 0.95):
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
                "filename": file.name,
                "num_qubits": qc.num_qubits,
                "depth": qc.depth(),
                "critical_depth": f.critical_depth,
                "parallelism": f.parallelism,
                "entanglement_ratio": f.entanglement_ratio,
            })
        except Exception:
            continue

    df = pd.DataFrame(features)

    if remove_depth_outliers:
        threshold = df["depth"].quantile(outlier_quantile)
        print(threshold)
        df = df[df["depth"] <= threshold]

    return df

# Load filtered features
df = load_all_features(remove_depth_outliers=True, outlier_quantile=0.99)

# Plot violin plots
metrics = ["num_qubits", "depth", "critical_depth", "parallelism", "entanglement_ratio"]
n_metrics = len(metrics)

fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
for ax, metric in zip(axes, metrics):
    sns.violinplot(y=df[metric], ax=ax, inner="box", palette="muted")
    ax.set_title(f"{metric}")
    ax.set_xlabel("")
    ax.set_ylabel("Value")
    ax.grid(True)

plt.suptitle("Distribution of Circuit Features (MQT Bench, depth outliers removed)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
