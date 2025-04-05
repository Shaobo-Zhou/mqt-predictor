import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gc
import csv
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit
from pathlib import Path
import networkx as nx
import numpy as np
from mqt.bench.utils import calc_supermarq_features

def get_path_training_circuits() -> Path:
    return Path(__file__).resolve().parent / "training_data" / "training_circuits"

def load_all_features_to_csv(max_qubits: int = None):
    base_path = get_path_training_circuits() / "mqt_bench_training"
    file_list = list(base_path.rglob("*.qasm"))
    
    output_path = Path(__file__).resolve().parent / "ig_circuit_complexity_metrics.csv"
    columns = [
        "file", "num_qubits", "depth", "gate_count",
        "program_communication", "parallelism", "entanglement_ratio", "liveness",
        "avg_hopcount", "max_degree", "min_degree", "adj_std"
    ]

    # Create CSV and write header
    with open(output_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()

    for count, file in enumerate(file_list):
        print(f"Processing circuit {count}: {file.name}")
        try:
            qc = QuantumCircuit.from_qasm_file(str(file))
            if max_qubits and qc.num_qubits > max_qubits:
                continue
            f = calc_supermarq_features(qc)

            # --- Build interaction graph ---
            G = nx.Graph()
            for gate in qc.data:
                if len(gate.qubits) == 2:
                    q0 = qc.qubits.index(gate.qubits[0])
                    q1 = qc.qubits.index(gate.qubits[1])
                    if G.has_edge(q0, q1):
                        G[q0][q1]["weight"] += 1
                    else:
                        G.add_edge(q0, q1, weight=1)

            if len(G.nodes) > 1 and nx.is_connected(G):
                avg_hopcount = nx.average_shortest_path_length(G)
                degrees = dict(G.degree())
                max_degree = max(degrees.values())
                min_degree = min(degrees.values())
                weights = [attr["weight"] for _, _, attr in G.edges(data=True)]
                adj_std = np.std(weights) if weights else 0
            else:
                avg_hopcount = np.nan
                max_degree = np.nan
                min_degree = np.nan
                adj_std = np.nan

            row = {
                "file": str(file),
                "num_qubits": qc.num_qubits,
                "depth": qc.depth,
                "gate_count": len(qc.data),
                "program_communication": f.program_communication,
                "parallelism": f.parallelism,
                "entanglement_ratio": f.entanglement_ratio,
                "liveness": f.liveness,
                "avg_hopcount": avg_hopcount,
                "max_degree": max_degree,
                "min_degree": min_degree,
                "adj_std": adj_std,
            }

            with open(output_path, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=columns)
                writer.writerow(row)

        except Exception as e:
            print(f"❌ Error in {file}: {e}")
        finally:
            del qc, G, f   
            gc.collect()

    print(f"\n✅ Done. Results saved to: {output_path}")

#load_all_features_to_csv()
""" # Load features
df = load_all_features()
output_path = Path(__file__).resolve().parent / "circuit_complexity_metrics.csv"
df.to_csv(output_path, index=False)

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
df.to_csv(output_path, index=False) """

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

#plot_complexity_distribution("complexity", "Distribution: Full Feature Complexity")
#plot_complexity_distribution("complexity_qubits_only", "Distribution: Qubit-Only Complexity")
#plot_complexity_distribution("complexity_qubits_gates", "Distribution: Qubit + Gate Complexity")


# --- Load data ---
output_path = Path(__file__).resolve().parent / "ig_circuit_complexity_metrics.csv"
df = pd.read_csv(output_path)

# --- Log-transform metrics that are skewed ---
df["log_gate_count"] = np.log1p(df["gate_count"])
df["log_avg_hopcount"] = np.log1p(df["avg_hopcount"])
df["log_adj_std"] = np.log1p(df["adj_std"])

# --- List of metrics to normalize ---
raw_features = [
    "num_qubits",
    "log_gate_count",
    "log_avg_hopcount",  # will invert after normalization
    "max_degree",
    "min_degree",
    "log_adj_std",       # optional to invert — see below
]

# --- Normalize ---
scaler = MinMaxScaler()
df_norm = pd.DataFrame(
    scaler.fit_transform(df[raw_features]),
    columns=[f"{col}_norm" for col in raw_features]
)

# --- Add normalized features to original df ---
df = pd.concat([df, df_norm], axis=1)

# --- Invert metrics where lower = more complex ---
df["log_avg_hopcount_norm_inv"] = 1 - df["log_avg_hopcount_norm"]
# Optional: Invert if high adj_std means *less* complexity
# df["log_adj_std_norm_inv"] = 1 - df["log_adj_std_norm"]
# Otherwise, use as-is
df["log_adj_std_norm_final"] = df["log_adj_std_norm"]

# --- Final feature list for complexity ---
final_features = [
    "num_qubits_norm",
    "log_gate_count_norm",
    "log_avg_hopcount_norm_inv",
    "max_degree_norm",
    "min_degree_norm",
    "log_adj_std_norm_final",
]

# --- Compute final complexity score ---
df["complexity_score"] = df[final_features].mean(axis=1)

# --- Bin into curriculum stages ---
df["complexity_bin"] = pd.qcut(
    df["complexity_score"], q=5,
    labels=["very_easy", "easy", "medium", "hard", "very_hard"]
)

# --- Plot the result ---
plot_complexity_distribution("complexity_score", "Distribution of Complexity Score (Inverted Metrics Included)")

""" # Define metrics for each row
metrics_row1 = ["num_qubits", "log_gate_count", "program_communication", "parallelism", "entanglement_ratio", "liveness"]
metrics_row2 = ["log_avg_hopcount", "max_degree", "min_degree", "log_adj_std"]

# Total number of subplots in each row
ncols = max(len(metrics_row1), len(metrics_row2))

# Create figure with 2 rows and ncols columns
fig, axes = plt.subplots(2, ncols, figsize=(6 * ncols, 10))

# Make sure axes is 2D even if ncols = 1
axes = axes.reshape(2, ncols)

# --- Plot Row 1 ---
for i, metric in enumerate(metrics_row1):
    ax = axes[0, i]
    sns.violinplot(y=df[metric].dropna(), ax=ax, inner="box", color="skyblue", cut=0, linewidth=1)
    ax.set_title(metric.replace("_", " ").title())
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    if metric not in {"num_qubits", "log_gate_count", "depth"}:
        ax.axhline(0, color="gray", linestyle="--", linewidth=1)
        ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        ax.set_ylim(-0.2, 1.2)

# --- Plot Row 2 ---
for i, metric in enumerate(metrics_row2):
    ax = axes[1, i]
    sns.violinplot(y=df[metric].dropna(), ax=ax, inner="box", color="skyblue", cut=0, linewidth=1)
    ax.set_title(metric.replace("_", " ").title())
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# --- Hide any unused subplots in Row 2 ---
for j in range(len(metrics_row2), ncols):
    fig.delaxes(axes[1, j])

plt.suptitle("Distribution of Circuit Features", fontsize=18)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.subplots_adjust(hspace=0.5, wspace=0.4)
plt.show() """