import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
file_path = "training_rewards.csv"
df = pd.read_csv(file_path)

# Apply log transform to depth and gate_count
#df["depth"] = np.log1p(df["depth"])  # log1p to avoid log(0)
#df["gate_count"] = np.log1p(df["gate_count"])

# Normalize all three metrics
df["num_qubits_norm"] = (df["num_qubits"] - df["num_qubits"].min()) / (df["num_qubits"].max() - df["num_qubits"].min())
df["depth_norm"] = (df["depth"] - df["depth"].min()) / (df["depth"].max() - df["depth"].min())
df["gate_count_norm"] = (df["gate_count"] - df["gate_count"].min()) / (df["gate_count"].max() - df["gate_count"].min())

#df.to_csv(file_path)

# Compute the average complexity score
#df["complexity_score"] = df[["num_qubits_norm", "depth_norm", "gate_count_norm"]].mean(axis=1)

# Bin into difficulty categories
labels = ["very_easy", "easy", "medium", "hard", "very_hard"]

#df["qubits_bin"] = pd.qcut(df["num_qubits"], q=5, labels=labels)

""" # Plot reward vs complexity score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="num_qubits", y="reward", hue="qubits_bin", palette="viridis", alpha=0.8)
plt.title("Reward vs. Num Qubits")
plt.xlabel("Number of qubits")
plt.ylabel("Reward")
plt.legend(title="Complexity Bin")
plt.grid(True)
plt.tight_layout()
plt.show() """

df["gate_count_bin"] = pd.qcut(df["gate_count"], q=5, labels=labels)
mean_rewards = df.groupby("gate_count_bin")[["gate_count", "reward"]].mean().reset_index()

# Plot scatter
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="gate_count", y="reward", hue="gate_count_bin", palette="viridis", alpha=0.6)
for _, row in mean_rewards.iterrows():
    x = row["gate_count"]
    y = row["reward"]
    label = f"{y:.3f}"  # Format to 3 decimal places
    plt.text(x + 1, y, label, fontsize=9, color="black", ha='left', va='center')
# Overlay mean points
sns.scatterplot(data=mean_rewards, x="gate_count", y="reward", color="black", marker="X", s=100, label="Bin Mean")

plt.title("Reward vs. Gate Count with Bin Averages")
plt.xlabel("Gate Count")
plt.ylabel("Reward")
plt.legend(title="Complexity Bin")
plt.grid(True)
plt.tight_layout()
plt.show()

""" # Plot reward vs complexity score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="gate_count", y="reward", hue="gate_count_bin", palette="viridis", alpha=0.8)
plt.title("Reward vs. Gate Count")
plt.xlabel("Gate Count")
plt.ylabel("Reward")
plt.legend(title="Complexity Bin")
plt.grid(True)
plt.tight_layout()
plt.show()


df["depth_bin"] = pd.qcut(df["depth"], q=5, labels=labels)

# Plot reward vs complexity score
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="depth", y="reward", hue="depth_bin", palette="viridis", alpha=0.8)
plt.title("Reward vs. Circuit Depth")
plt.xlabel("Circuit Depth")
plt.ylabel("Reward")
plt.legend(title="Complexity Bin")
plt.grid(True)
plt.tight_layout()
plt.show() """
