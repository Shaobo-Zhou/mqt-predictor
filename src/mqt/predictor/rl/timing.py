import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the saved action timing data
with open("action_timings.json", "r") as f:
    timings = json.load(f)

# Normalize: ensure each value is a list (even if someone accidentally saved a float)
normalized_timings = {
    action: times if isinstance(times, list) else [times]
    for action, times in timings.items()
}

# Compute average timing per action
df = pd.DataFrame([
    {"action": action, "avg_time": sum(times) / len(times)}
    for action, times in normalized_timings.items() if times
])

# Sort by average timing (ascending)
df_sorted = df.sort_values(by="avg_time")

# ✅ Print the average time per action
print("📊 Average execution time per action:")
for _, row in df_sorted.iterrows():
    print(f"🔹 {row['action']}: {row['avg_time']:.4f} seconds")

# ✅ Plotting
plt.figure(figsize=(12, 6))
plt.barh(df_sorted["action"], df_sorted["avg_time"])
plt.xlabel("Average Time (seconds)")
plt.ylabel("Action")
plt.title("Average Execution Time per Action (Sorted)")
plt.tight_layout()
plt.show()

