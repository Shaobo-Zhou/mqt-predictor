import pandas as pd
from pathlib import Path
from qiskit import QuantumCircuit
from importlib import resources
import shutil

# Load training bin ranges
training_df = pd.read_csv("curriculum_metrics_gate_count.csv")
bin_ranges = training_df.groupby("complexity_bin")["gate_count"].agg(["min", "max"]).sort_index().reset_index()

# Load test circuits
data_path = Path(str(resources.files("mqt.predictor"))) / "rl" / "training_data"
test_dir = data_path / "training_circuits" / "test_data_compilation"
test_qasm_files = list(test_dir.rglob("*.qasm"))

test_data = []
for qasm_path in test_qasm_files:
    qc = QuantumCircuit.from_qasm_file(str(qasm_path))
    test_data.append({"file": str(qasm_path), "gate_count": qc.size()})

test_df = pd.DataFrame(test_data)

# Assign bins
def assign_bin(gate_count):
    for _, row in bin_ranges.iterrows():
        if row["min"] <= gate_count <= row["max"]:
            return row["complexity_bin"]
    return "ignore"

# Make a copy to avoid side effects
test_df = test_df.copy()
test_df["complexity_bin"] = test_df["gate_count"].apply(assign_bin)

# Sample 10 from each valid bin
sampled_df = test_df[test_df["complexity_bin"] != "ignore"]
sampled_df = (
    sampled_df.groupby("complexity_bin", group_keys=False)
    .apply(lambda x: x.sample(n=min(5, len(x)), random_state=42))
    .reset_index(drop=True)
)

# Create the new directory under the same parent as test_data_compilation
representative_dir = test_dir.parent / "representative_test_data"
representative_dir.mkdir(exist_ok=True)
sampled_df_path = representative_dir / "sampled_representative_circuits.csv"
sampled_df = sampled_df.sort_values(by="gate_count").reset_index(drop=True)
sampled_df.to_csv(sampled_df_path, index=False)

print(f"✅ Saved sampled DataFrame to: {sampled_df_path}")


# Copy the sampled .qasm files
for file_path in sampled_df["file"]:
    destination = representative_dir / Path(file_path).name
    shutil.copy(file_path, destination)

print(f"✅ Copied {len(sampled_df)} files to: {representative_dir}")
