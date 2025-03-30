import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from qiskit import QuantumCircuit
from pathlib import Path
from mqt.bench.utils import calc_supermarq_features

def get_path_training_circuits() -> Path:
    return Path(__file__).resolve().parent / "training_data" / "training_circuits"

def load_all_features(max_qubits: int = None):
    base_path = get_path_training_circuits() / "mqt_bench_training"
    file_list = list(base_path.rglob("*.qasm"))  # recursively find all .qasm files

    features = []
    for file in file_list:
        try:
            qc = QuantumCircuit.from_qasm_file(str(file))
            if max_qubits and qc.num_qubits > max_qubits:
                continue
            f = calc_supermarq_features(qc)
            features.append({
                "num_qubits": qc.num_qubits,
                "depth": qc.depth(),
                "critical_depth": f.critical_depth,
                "parallelism": f.parallelism,
                "entanglement_ratio": f.entanglement_ratio,
            })
        except Exception:
            continue
    return features

# Load and prepare data
features = load_all_features()
df = pd.DataFrame(features)

# Melt the DataFrame to long-form for seaborn
df_melted = df.melt(var_name="Metric", value_name="Value")

# Plot violin plots
plt.figure(figsize=(16, 6))
sns.violinplot(x="Metric", y="Value", data=df_melted, inner="box", palette="muted")
plt.title("Distribution of Circuit Features (MQT Bench)")
plt.ylabel("Value")
plt.xlabel("Feature")
plt.grid(True)
plt.tight_layout()
plt.show()
