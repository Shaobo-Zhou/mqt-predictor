import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from pathlib import Path
import zipfile
import os
from mqt.bench.utils import calc_supermarq_features
from mqt.predictor import rl


def get_path_training_circuits() -> Path:
    return Path(__file__).resolve().parent / "training_data" / "training_circuits"

def load_all_features(max_qubits: int = None):
    path = get_path_training_circuits()
    file_list = list(path.glob("*.qasm"))

    zip_path = path / "MQTBench.zip"
    if zip_path.exists():
        with zipfile.ZipFile(str(zip_path), "r") as zip_ref:
            zip_ref.extractall(path)
        file_list = list(path.glob("*.qasm"))

    features = []
    for file in file_list:
        try:
            num_qubits = int(file.stem.split("_")[-1])
            if max_qubits and num_qubits > max_qubits:
                continue
            qc = QuantumCircuit.from_qasm_file(str(file))
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

features = load_all_features()

# Plot histograms
fig, axs = plt.subplots(1, 5, figsize=(25, 5))
metrics = ["num_qubits", "depth", "critical_depth", "parallelism", "entanglement_ratio"]

for ax, metric in zip(axs, metrics):
    values = [f[metric] for f in features]
    ax.hist(values, bins=20, edgecolor='black')
    ax.set_title(f"Distribution of {metric}")
    ax.set_xlabel(metric)
    ax.set_ylabel("Count")

plt.tight_layout()
plt.show()
